#!/bin/bash
#
# This script cleans up all of the build data

set -e

WS_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
UNITREE_WS="$WS_ROOT/src/third_party/unitree_ros2/cyclonedds_ws"

echo "--- Cleaning Unitree build ---"
cd "$UNITREE_WS"
rm -rf build/ install/ log/

echo "--- Cleaning main workspace build ---"
cd "$WS_ROOT"
rm -rf build/ install/ log/

echo "Done. Please use scripts/build_all.sh to re-build"
