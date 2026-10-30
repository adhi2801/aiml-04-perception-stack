"""AEB decision node: /perception/objects -> /aeb/command, with a fail-safe watchdog.

If perception goes silent for longer than `watchdog_s` (crash, GPU stall, dropped link), the node
commands BRAKE: in a safety system, losing perception must never look like "road is clear".
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import Int8, String

ROOT = Path(os.environ.get("PERCEPTION_ROOT", Path.home() / "aiml-04-perception-stack"))
sys.path.insert(0, str(ROOT / "src"))

from perception.fusion.aeb import AEB, Action, AEBConfig, TrackedObject  # noqa: E402


class AEBNode(Node):
    def __init__(self):
        super().__init__("aeb")
        cfg = yaml.safe_load((ROOT / "configs" / "pipeline.yaml").read_text())["aeb"]
        self.aeb = AEB(AEBConfig.from_dict(cfg))
        self.watchdog_s = self.declare_parameter("watchdog_s", 0.5).value
        self.last_msg = time.monotonic()
        self.pub_cmd = self.create_publisher(Int8, "/aeb/command", 10)
        self.pub_status = self.create_publisher(String, "/aeb/status", 10)
        self.create_subscription(String, "/perception/objects", self.on_objects, 10)
        self.create_timer(0.1, self.watchdog)

    def on_objects(self, msg: String):
        self.last_msg = time.monotonic()
        data = json.loads(msg.data)
        objs = [TrackedObject(o["id"], o["cls"], tuple(o["box"]), o["distance_m"], o["conf"]) for o in data["objects"]]
        corridor_fn = None
        if len(data["corridor"]) >= 2:
            c = np.array(data["corridor"])
            corridor_fn = lambda y: (float(np.interp(y, c[:, 0], c[:, 1])), float(np.interp(y, c[:, 0], c[:, 2])))  # noqa: E731
        action, threat = self.aeb.step(data["t"], objs, data["img_w"], corridor_fn)
        self.pub_cmd.publish(Int8(data=int(action)))
        self.pub_status.publish(String(data=json.dumps({
            "action": action.name,
            "threat": None if threat is None else {"id": threat.track_id, "cls": threat.cls,
                                                   "distance_m": round(threat.distance_m, 2),
                                                   "ttc_s": None if threat.ttc_s is None else round(threat.ttc_s, 2)}})))
        if action == Action.BRAKE:
            self.get_logger().warn(f"BRAKE: {threat}", throttle_duration_sec=1.0)

    def watchdog(self):
        if time.monotonic() - self.last_msg > self.watchdog_s:
            self.pub_cmd.publish(Int8(data=int(Action.BRAKE)))
            self.get_logger().error("perception heartbeat lost -> fail-safe BRAKE", throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = AEBNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
