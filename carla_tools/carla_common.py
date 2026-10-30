"""Shared CARLA helpers: synchronous world, ego + sensors, ground-truth depth and 2D boxes."""
from __future__ import annotations

import math
import queue
from contextlib import contextmanager
from dataclasses import dataclass

import carla
import numpy as np


@dataclass
class CamSpec:
    width: int = 1280
    height: int = 720
    fov: float = 90.0
    x: float = 1.5     # meters forward of the vehicle origin
    z: float = 1.5     # mount height (matches configs/pipeline.yaml)


def intrinsics(spec: CamSpec) -> np.ndarray:
    f = spec.width / (2.0 * math.tan(math.radians(spec.fov) / 2.0))
    return np.array([[f, 0, spec.width / 2], [0, f, spec.height / 2], [0, 0, 1]])


@contextmanager
def sync_world(client: carla.Client, fps: float = 20.0, town: str | None = None):
    """Deterministic fixed-step simulation; settings are restored even if the script crashes."""
    world = client.load_world(town) if town else client.get_world()
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(settings)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    actors: list[carla.Actor] = []
    try:
        yield world, tm, actors
    finally:
        for a in actors:
            if a.is_alive:
                if isinstance(a, carla.Sensor):
                    a.stop()
                a.destroy()
        tm.set_synchronous_mode(False)
        world.apply_settings(original)


def spawn_ego(world, actors, spawn_index: int = 0, blueprint: str = "vehicle.tesla.model3") -> carla.Vehicle:
    bp = world.get_blueprint_library().find(blueprint)
    bp.set_attribute("role_name", "hero")
    ego = world.spawn_actor(bp, world.get_map().get_spawn_points()[spawn_index])
    actors.append(ego)
    return ego


def attach_camera(world, actors, ego, spec: CamSpec, kind: str = "rgb") -> tuple[carla.Sensor, queue.Queue]:
    bp = world.get_blueprint_library().find({"rgb": "sensor.camera.rgb", "depth": "sensor.camera.depth",
                                             "semantic": "sensor.camera.semantic_segmentation"}[kind])
    bp.set_attribute("image_size_x", str(spec.width))
    bp.set_attribute("image_size_y", str(spec.height))
    bp.set_attribute("fov", str(spec.fov))
    cam = world.spawn_actor(bp, carla.Transform(carla.Location(x=spec.x, z=spec.z)), attach_to=ego)
    q: queue.Queue = queue.Queue()
    cam.listen(q.put)
    actors.append(cam)
    return cam, q


def attach_collision(world, actors, ego) -> list:
    events: list = []
    s = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"), carla.Transform(), attach_to=ego)
    s.listen(lambda e: events.append((e.frame, e.other_actor.type_id)))
    actors.append(s)
    return events


def get_frame(q: queue.Queue, frame: int, timeout: float = 5.0):
    """Pop until the sensor image for this exact simulation frame arrives (sensors can lag a tick)."""
    while True:
        img = q.get(timeout=timeout)
        if img.frame == frame:
            return img


def rgb_to_bgr(img: carla.Image) -> np.ndarray:
    return np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[..., :3].copy()


def depth_to_meters(img: carla.Image) -> np.ndarray:
    bgra = np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4).astype(np.float32)
    normalized = (bgra[..., 2] + bgra[..., 1] * 256 + bgra[..., 0] * 65536) / (256 ** 3 - 1)
    return 1000.0 * normalized


def gt_boxes(world, camera: carla.Sensor, ego: carla.Actor, spec: CamSpec, depth_m: np.ndarray,
             max_dist: float = 60.0) -> list[dict]:
    """Projected 2D boxes for visible vehicles and walkers, with actor id and true distance.

    Occlusion check: at least 10% of the box must have depth within ~bbox extent of the actor distance.
    """
    K = intrinsics(spec)
    w2c = np.array(camera.get_transform().get_inverse_matrix())
    cam_loc = camera.get_transform().location
    fwd = camera.get_transform().get_forward_vector()
    out = []
    for actor in world.get_actors():
        if actor.id == ego.id or not (actor.type_id.startswith("vehicle.") or actor.type_id.startswith("walker.")):
            continue
        loc = actor.get_transform().location
        dist = loc.distance(cam_loc)
        to = loc - cam_loc
        if dist > max_dist or (to.x * fwd.x + to.y * fwd.y + to.z * fwd.z) <= 0:
            continue
        verts = actor.bounding_box.get_world_vertices(actor.get_transform())
        pts = np.array([[v.x, v.y, v.z, 1.0] for v in verts]).T
        cam = w2c @ pts
        cam = np.stack([cam[1], -cam[2], cam[0]])  # UE4 (x fwd, y right, z up) -> camera (x right, y down, z fwd)
        if (cam[2] <= 0.1).any():
            continue
        uv = (K @ cam)[:2] / cam[2]
        x1, y1 = np.clip(uv.min(axis=1), 0, [spec.width - 1, spec.height - 1])
        x2, y2 = np.clip(uv.max(axis=1), 0, [spec.width - 1, spec.height - 1])
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            continue
        patch = depth_m[int(y1):int(y2), int(x1):int(x2)]
        extent = max(actor.bounding_box.extent.x, actor.bounding_box.extent.y) + 1.0
        if np.mean(np.abs(patch - float(np.median(cam[2]))) < extent) < 0.1:
            continue  # occluded
        cls = "pedestrian" if actor.type_id.startswith("walker.") else _vehicle_class(actor)
        out.append({"id": actor.id, "cls": cls, "box": [float(x1), float(y1), float(x2), float(y2)],
                    "distance_m": float(np.min(cam[2]))})
    return out


def _vehicle_class(actor) -> str:
    base = actor.attributes.get("base_type", "").lower()
    return {"truck": "truck", "bus": "bus", "motorcycle": "motorcycle", "bicycle": "bicycle"}.get(base, "car")
