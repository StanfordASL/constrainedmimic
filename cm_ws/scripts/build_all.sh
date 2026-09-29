#!/bin/bash
#
# This script builds all ros2 packages, starting with the unitree ros2 underlay.

set -e

# Make sure this is run from a fresh terminal
if [[ -n "$ROS_DISTRO" ]]; then
  echo "ERROR: A ROS environment is already sourced."
  echo "Detected ROS_DISTRO=${ROS_DISTRO}"
  echo "Please run this script from a fresh terminal."
  exit 1
fi

WS_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
UNITREE_WS="$WS_ROOT/src/third_party/unitree_ros2/cyclonedds_ws"
CONSTRAINEDMIMIC_DIR="$WS_ROOT/.."

if [[ ! -d "$CONSTRAINEDMIMIC_DIR/.venv" ]]; then
  echo "ERROR: Could not find the constrainedmimic virtual env"
  echo "Check that you've set up the venv with UV"
  exit 1
fi

# Source the venv before building
source "$CONSTRAINEDMIMIC_DIR/.venv/bin/activate"
PYTHON_EXECUTABLE=$(which python3)
echo "Using Python: $PYTHON_EXECUTABLE"

echo "--- Building Unitree Underlay (Middleware & APIs) ---"
cd "$UNITREE_WS"
colcon build --packages-select cyclonedds
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install #--cmake-args -DCMAKE_BUILD_TYPE=Release --packages-ignore cyclonedds

echo "--- Sourcing Unitree Underlay ---"
source "$UNITREE_WS/install/setup.bash"

echo "--- Building Main Workspace ---"
cd "$WS_ROOT"

# We ignore the packages already built in the underlay to prevent conflicts
colcon build --symlink-install \
    --cmake-args -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
    --packages-ignore \
        cyclonedds \
        rmw_cyclonedds_cpp \
        unitree_api \
        unitree_go \
        unitree_hg

echo "Build complete. Use 'source scripts/setup_sim.sh' or 'source scripts/setup_hardware.sh' to refresh your environment."
