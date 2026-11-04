"""CARLA <-> ROS2 bridge: owns the simulator clock, publishes camera frames, applies AEB commands.

Publishes  /carla/ego/rgb (bgr8), /carla/ego/depth (32FC1, meters, ground truth), /carla/ego/speed (m/s)
Subscribes /aeb/command (Int8: 0 cruise, 1 warn, 2 brake)

The ego drives on Traffic Manager autopilot that ignores pedestrians and vehicles, so only
/aeb/command can stop it. A brake command takes control away from autopilot until the threat clears.
"""
import os
import random
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int8

from .img import to_msg

ROOT = Path(os.environ.get("PERCEPTION_ROOT", Path.home() / "aiml-04-perception-stack"))
sys.path.insert(0, str(ROOT / "carla_tools"))

import carla  # noqa: E402
from carla_common import (CamSpec, attach_camera, attach_collision, depth_to_meters, get_frame,  # noqa: E402
                          rgb_to_bgr, spawn_ego)


class CarlaBridge(Node):
    def __init__(self):
        super().__init__("carla_bridge")
        p = {k: self.declare_parameter(k, v).value for k, v in {
            "host": os.environ.get("CARLA_HOST", "localhost"), "port": 2000, "fps": 20.0, "spawn_index": 0,
            "scenario": "cpna25", "width": 1280, "height": 720, "fov": 90.0, "seed": 0}.items()}
        self.client = carla.Client(p["host"], p["port"])
        self.client.set_timeout(30.0)
        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()
        s = self.world.get_settings()
        s.synchronous_mode, s.fixed_delta_seconds = True, 1.0 / p["fps"]
        self.world.apply_settings(s)
        self.tm = self.client.get_trafficmanager()
        self.tm.set_synchronous_mode(True)

        self.actors = []
        spec = CamSpec(p["width"], p["height"], p["fov"])
        self.ego = spawn_ego(self.world, self.actors, p["spawn_index"])
        self.rgb_cam, self.rgb_q = attach_camera(self.world, self.actors, self.ego, spec, "rgb")
        _, self.depth_q = attach_camera(self.world, self.actors, self.ego, spec, "depth")
        self.collisions = attach_collision(self.world, self.actors, self.ego)
        self.ego.set_autopilot(True, self.tm.get_port())
        self.tm.ignore_walkers_percentage(self.ego, 100)
        self.tm.ignore_vehicles_percentage(self.ego, 100)
        self.world.tick()

        self.hazard_tick = None
        if p["scenario"] != "none":
            from run_scenarios import setup_hazard
            _, self.hazard_tick = setup_hazard(self.world, self.actors, self.ego, p["scenario"], random.Random(p["seed"]),
                                               self.tm)

        self.command, self.braking, self.step = 0, False, 0
        self.pub_rgb = self.create_publisher(Image, "/carla/ego/rgb", qos_profile_sensor_data)
        self.pub_depth = self.create_publisher(Image, "/carla/ego/depth", qos_profile_sensor_data)
        self.pub_speed = self.create_publisher(Float32, "/carla/ego/speed", 10)
        self.create_subscription(Int8, "/aeb/command", self.on_command, 10)
        self.create_timer(1.0 / p["fps"], self.tick)
        self.get_logger().info(f"connected to CARLA {p['host']}:{p['port']}, scenario={p['scenario']}")

    def on_command(self, msg: Int8):
        self.command = msg.data

    def tick(self):
        frame = self.world.tick()
        self.step += 1
        if self.hazard_tick:
            self.hazard_tick(self.step)
        stamp = self.get_clock().now().to_msg()
        rgb = to_msg(rgb_to_bgr(get_frame(self.rgb_q, frame)), "bgr8")
        depth = to_msg(depth_to_meters(get_frame(self.depth_q, frame)), "32FC1")
        for m in (rgb, depth):
            m.header.stamp, m.header.frame_id = stamp, "ego_camera"
        self.pub_rgb.publish(rgb)
        self.pub_depth.publish(depth)
        self.pub_speed.publish(Float32(data=float(self.ego.get_velocity().length())))

        if self.command == 2:
            if not self.braking:
                self.ego.set_autopilot(False, self.tm.get_port())
                self.braking = True
                self.get_logger().warn("AEB BRAKE")
            self.ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        elif self.braking:
            self.ego.set_autopilot(True, self.tm.get_port())
            self.braking = False
            self.get_logger().info("AEB released, autopilot resumed")
        if self.collisions:
            self.get_logger().error(f"COLLISION with {self.collisions[-1][1]}")
            self.collisions.clear()

    def destroy_node(self):
        for a in reversed(self.actors):
            if a.is_alive:
                if isinstance(a, carla.Sensor):
                    a.stop()
                a.destroy()
        self.tm.set_synchronous_mode(False)
        self.world.apply_settings(self.original_settings)
        super().destroy_node()


def main():
    rclpy.init()
    node = CarlaBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
