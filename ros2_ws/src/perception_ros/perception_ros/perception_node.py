"""Perception node: camera frame -> tracked objects with distance + ego-lane corridor.

Subscribes /carla/ego/rgb (and /carla/ego/depth when oracle_depth:=true)
Publishes  /perception/detections   vision_msgs/Detection2DArray  (standard, for RViz/foxglove)
           /perception/objects      std_msgs/String JSON (track id, class, box, distance, corridor samples)
           /perception/debug_image  sensor_msgs/Image (annotated)
           /perception/latency_ms   std_msgs/Float32
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .img import from_msg, to_msg

ROOT = Path(os.environ.get("PERCEPTION_ROOT", Path.home() / "aiml-04-perception-stack"))
sys.path.insert(0, str(ROOT / "src"))

from perception.pipeline import PerceptionPipeline  # noqa: E402

CORRIDOR_ROWS = np.linspace(380, 719, 12)


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception")
        self.oracle = self.declare_parameter("oracle_depth", False).value
        cfg = self.declare_parameter("config", str(ROOT / "configs" / "pipeline.yaml")).value
        os.chdir(ROOT)  # model paths in the config are repo-relative
        self.pipe = PerceptionPipeline(cfg, use_depth_model=not self.oracle)
        self.latest_depth = None
        self.create_subscription(Image, "/carla/ego/rgb", self.on_rgb, qos_profile_sensor_data)
        if self.oracle:
            self.create_subscription(Image, "/carla/ego/depth", self.on_depth, qos_profile_sensor_data)
        self.pub_det = self.create_publisher(Detection2DArray, "/perception/detections", 10)
        self.pub_obj = self.create_publisher(String, "/perception/objects", 10)
        self.pub_dbg = self.create_publisher(Image, "/perception/debug_image", qos_profile_sensor_data)
        self.pub_lat = self.create_publisher(Float32, "/perception/latency_ms", 10)
        self.get_logger().info(f"perception ready (weights={self.pipe.cfg['detector']['weights']}, oracle_depth={self.oracle})")

    def on_depth(self, msg: Image):
        self.latest_depth = from_msg(msg)

    def on_rgb(self, msg: Image):
        t0 = time.perf_counter()
        bgr = from_msg(msg)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        res = self.pipe(bgr, t=t, gt_depth=self.latest_depth if self.oracle else None)

        arr = Detection2DArray(header=msg.header)
        for o in res.objects:
            d = Detection2D(header=msg.header, id=str(o.track_id))
            x1, y1, x2, y2 = o.box
            d.bbox.center.position.x, d.bbox.center.position.y = (x1 + x2) / 2, (y1 + y2) / 2
            d.bbox.size_x, d.bbox.size_y = x2 - x1, y2 - y1
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = o.cls, o.conf
            d.results.append(hyp)
            arr.detections.append(d)
        self.pub_det.publish(arr)

        corridor = []
        if res.lanes.valid:
            for y in CORRIDOR_ROWS:
                c = self.pipe.lanes.corridor_at(float(y))
                if c:
                    corridor.append([float(y), *c])
        self.pub_obj.publish(String(data=json.dumps({
            "t": t, "img_w": bgr.shape[1],
            "objects": [{"id": o.track_id, "cls": o.cls, "box": o.box, "distance_m": o.distance_m, "conf": o.conf}
                        for o in res.objects],
            "corridor": corridor, "timings_ms": res.timings_ms})))
        self.pub_dbg.publish(to_msg(self.pipe.draw(bgr, res), "bgr8", msg.header))
        self.pub_lat.publish(Float32(data=(time.perf_counter() - t0) * 1000))


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
