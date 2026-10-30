"""sensor_msgs/Image <-> numpy without cv_bridge (avoids its NumPy-version coupling)."""
import numpy as np
from sensor_msgs.msg import Image

_DTYPES = {"bgr8": (np.uint8, 3), "rgb8": (np.uint8, 3), "mono8": (np.uint8, 1), "32FC1": (np.float32, 1)}


def to_msg(arr: np.ndarray, encoding: str, header=None) -> Image:
    msg = Image()
    if header is not None:
        msg.header = header
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = arr.strides[0]
    msg.data = np.ascontiguousarray(arr).tobytes()
    return msg


def from_msg(msg: Image) -> np.ndarray:
    dtype, ch = _DTYPES[msg.encoding]
    arr = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, ch) if ch > 1 else \
        np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
    return arr[..., ::-1].copy() if msg.encoding == "rgb8" else arr.copy()
