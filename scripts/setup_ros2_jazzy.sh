#!/usr/bin/env bash
# One-shot setup inside the WSL2 Ubuntu 24.04 distro: ROS2 Jazzy + this project's Python deps.
# Idempotent: safe to re-run.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo sudo)

$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends software-properties-common curl ca-certificates gnupg locales \
    python3-pip python3-venv git
$SUDO locale-gen en_US.UTF-8
$SUDO add-apt-repository -y universe

# ROS2 apt source (official ros2-apt-source package keeps the signing key current)
if ! dpkg -s ros2-apt-source >/dev/null 2>&1; then
  VER=$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -oP '"tag_name": "\K[^"]+')
  curl -fsSL -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${VER}/ros2-apt-source_${VER}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb"
  $SUDO dpkg -i /tmp/ros2-apt-source.deb
fi
$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends ros-jazzy-ros-base ros-jazzy-vision-msgs \
    python3-colcon-common-extensions libgl1 libglib2.0-0

grep -q "/opt/ros/jazzy/setup.bash" ~/.bashrc || echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

# Python deps in a venv that can still see ROS's system packages (rclpy lives there).
ROOT=$(cd "$(dirname "$0")/.." && pwd)
python3 -m venv --system-site-packages ~/venvs/perception
# shellcheck disable=SC1090
source ~/venvs/perception/bin/activate
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -c "$ROOT/constraints-cu124.txt" --extra-index-url https://download.pytorch.org/whl/cu124 \
    "ultralytics>=8.3" opencv-python-headless pyyaml "timm>=1.0" "tensorrt-cu12>=10.3,<11" "onnxruntime-gpu>=1.19" \
    wandb carla==0.9.16
pip install --no-deps -e "$ROOT"
grep -q "venvs/perception" ~/.bashrc || echo "source ~/venvs/perception/bin/activate" >> ~/.bashrc
echo "done: open a new shell, then: cd $ROOT/ros2_ws && colcon build --symlink-install"
