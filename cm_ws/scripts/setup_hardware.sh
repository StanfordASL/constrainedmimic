#!/bin/bash
#
# This script configures the environment for working with the Unitree hardware.

# JAX configuration
export JAX_PLATFORMS="cpu"
export JAX_ENABLE_X64=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export OPENBLAS_NUM_THREADS=1

WS_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

echo "Configuring environment for HARDWARE..."

# 1. Path to the Unitree internal workspace
UNITREE_WS="$WS_ROOT/src/third_party/unitree_ros2/cyclonedds_ws"
CONSTRAINEDMIMIC_DIR="$WS_ROOT/.."

if [[ ! -d "$CONSTRAINEDMIMIC_DIR/.venv" ]]; then
  echo "ERROR: Could not find constrainedmimic virtual env"
  echo "Check that you've set up the venv with UV"
  return 1
fi

export ROS_DOMAIN_ID=0
echo "Using ROS_DOMAIN_ID=$ROS_DOMAIN_ID"

# Source the venv for the project
source "$CONSTRAINEDMIMIC_DIR/.venv/bin/activate"

# 2. Source the Unitree Underlay
if [ -f "$UNITREE_WS/install/setup.bash" ]; then
    source "$UNITREE_WS/install/setup.bash"
    echo "✓ Unitree underlay sourced."
else
    echo "X Warning: Unitree underlay not built yet. Run 'colcon build' in $UNITREE_WS"
fi

# 3. Source your Main Workspace
if [ -f "$WS_ROOT/install/setup.bash" ]; then
    source "$WS_ROOT/install/setup.bash"
    echo "✓ Main workspace sourced."
else
    echo "! Main workspace not built yet."
fi

# 4. MuJoCo Library Path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/mujoco/lib

# 4. CycloneDDS Configuration (Hardware)
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
                            <NetworkInterface name="enp2s0" priority="default" multicast="default" />
                        </Interfaces></General></Domain></CycloneDDS>'

echo "✓ Unitree ROS2 environment configured for HARDWARE (Robot: enp2s0)"
