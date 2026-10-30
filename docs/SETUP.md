# Setup: Windows 11 + WSL2 (tested on a GTX 1650 4 GB laptop)

```
Windows host                                   WSL2 "ros2-jazzy" (Ubuntu 24.04, NAT networking)
┌──────────────────────────┐   TCP 2000-2002   ┌───────────────────────────────────────┐
│ CARLA 0.9.16 server      │◄─────────────────►│ ROS2 Jazzy                            │
│ (renders on the GPU)     │  via host IP      │  carla_bridge → perception → aeb      │
└──────────────────────────┘                   │  (YOLO TensorRT + MiDaS on the GPU)   │
                                               └───────────────────────────────────────┘
```
Ubuntu 24.04 + ROS2 Jazzy (LTS to 2029) both use Python 3.12, which matches the CARLA 0.9.16 client wheel
and the Windows environment, so one set of dependency versions works everywhere.

> WSL's global "mirrored networking" mode is deliberately **not** used: it changes every WSL distro, including
> Docker Desktop's, and can break existing container networking. NAT plus the host IP only affects this distro.

## 1. CARLA server (Windows)
Extract `CARLA_0.9.16.zip`, then start it with low settings so CARLA and the models share 4 GB of VRAM:
```powershell
.\CARLA_0.9.16\CarlaUE4.exe -quality-level=Low -RenderOffScreen -carla-rpc-port=2000
```
The first start triggers a Windows Firewall prompt. Allow **private networks**, since WSL reaches CARLA through it.
Drop `-RenderOffScreen` to watch the spectator view (costs ~0.5 GB VRAM).

## 2. Direct mode (no ROS, Windows only)
For batch evaluation and the demo video, everything runs from the Windows venv:
```powershell
.venv\Scripts\python carla_tools\run_scenarios.py --scenario pedestrian_crossing --trials 10 --video reports\demo_pedestrian.mp4
```

## 3. WSL2 distro with ROS2 Jazzy
```powershell
# official Ubuntu 24.04 WSL image (checksum-verified), imported as its own distro
wsl --import ros2-jazzy C:\path\to\capstone\wsl\ros2-jazzy ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz --version 2
wsl -d ros2-jazzy
```
Inside the distro:
```bash
bash scripts/setup_ros2_jazzy.sh          # ROS2 Jazzy + vision_msgs + colcon + Python deps (CUDA torch, TensorRT)
source ~/.bashrc
nvidia-smi                                # the Windows driver exposes the GPU to WSL automatically
```

## 4. Build and launch
```bash
cd ros2_ws && colcon build --symlink-install && source install/setup.bash
export PERCEPTION_ROOT=$(realpath ..)
export CARLA_HOST=$(ip route show default | awk '{print $3}')   # the Windows host as seen from WSL
ros2 launch perception_ros perception.launch.py scenario:=pedestrian_crossing
# another terminal:
ros2 topic hz /perception/objects ; ros2 topic echo /aeb/status
```

## Troubleshooting
| Symptom | Fix |
|---|---|
| `time-out ... waiting for the simulator` from WSL | CARLA not running, `CARLA_HOST` not set, or the firewall prompt was denied (re-allow `CarlaUE4-Win64-Shipping.exe` on private networks) |
| CUDA out of memory | Use `-quality-level=Low`, the TensorRT engine, `MiDaS_small`, and close the spectator window |
| The car brakes immediately at start | Expected: the AEB watchdog fail-safe brakes until the first perception message (model load ~5 s) |
| Remove everything afterwards | `wsl --unregister ros2-jazzy` deletes the distro and its disk image |
