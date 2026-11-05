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

### Running CARLA on a 4 GB GPU / 16 GB RAM laptop (what actually worked)
Measured on a GTX 1650 4 GB, i7-11370H, 16 GB RAM:

| Launch | Result |
|---|---|
| `-quality-level=Low -RenderOffScreen` (default flags) | allocated 2.7 GB VRAM, then the process vanished during map load, with no log |
| `+ -windowed -ResX=640 -ResY=360` | 3.8 GB VRAM + 4.8 GB RAM at 20 s, then exited |
| `+ -dx11 -ini:Engine:[/Script/Engine.RendererSettings]:r.Streaming.PoolSize=300` | **stable**: ~3.4 GB RAM, 3.9 GB VRAM |

Root cause: the Windows **commit limit**. Once VRAM is full, the driver spills into shared system memory,
which counts against the same budget as every open app. With a browser, an IDE, Steam and Docker
Desktop open, ~23 of 33 GB was already committed, so CARLA (and later Python + YOLO) pushed it over and the
OS killed the process before Unreal could write a crash log.

That recipe keeps an idle server alive, but **cameras did not work on it** (measured 2026-11-05, NVIDIA driver 610.91):

| Symptom | Cause | Fix |
|---|---|---|
| With DX11, a camera delivers ~5 frames, then the server stops answering; repeated `EXCEPTION_ACCESS_VIOLATION` crash reports | Unreal 4.26's DX11 renderer with this driver | **`-dx12`**: steady ~63 FPS from a 640×360 camera over 20 s |
| `streaming client: connection failed ... refused` | `localhost` can resolve to `::1`; CARLA listens on IPv4 only | connect to `127.0.0.1` (the scripts' default) |
| Town10HD idle used 3.9 GB VRAM and left ~2 GB of commit; switching maps crashed it | Town10HD is the heaviest map, and a map switch briefly holds both | start directly on **Town03_Opt** (set `GameDefaultMap`/`ServerDefaultMap` in `CarlaUE4/Config/DefaultEngine.ini`): 1.9 GB VRAM idle |

With those three changes, RGB + depth + semantic cameras at 800×450 plus 20 autopilot vehicles run at
20 synchronous ticks per second (real time) using 2.5 GB of VRAM.

Recipe that runs:
```powershell
.\CARLA_0.9.16\CarlaUE4.exe -quality-level=Low -dx12 -windowed -ResX=320 -ResY=180 -nosound `
  "-ini:Engine:[/Script/Engine.RendererSettings]:r.Streaming.PoolSize=300"
```
- Close browsers, IDEs and game launchers first (aim for **≥ 8 GB free commit**:
  `(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB`).
- Use `configs/pipeline_lowvram.yaml` (800×450 cameras).
- CARLA leaves ~200 MB of VRAM, so run detection on the CPU (`$env:CUDA_VISIBLE_DEVICES=""`) or use the TensorRT
  engine with a small workspace. Report FPS for the setup you actually used.

## 2. Direct mode (no ROS, Windows only)
For batch evaluation and the demo video, everything runs from the Windows venv:
```powershell
.venv\Scripts\python carla_tools\run_scenarios.py --scenario cpna25 --speeds 20,30,40 --trials 5 --video reports\demo_cpna25.mp4
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
ros2 launch perception_ros perception.launch.py scenario:=cpna25
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
