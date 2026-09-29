# Assorted notes (WIP)

These notes were taken from some of the READMEs before combining the repos together. I didn't want to delete them, so putting them here for now. We'll clean this up later

## Pico

Running the node: `ros2 run xrobo_ros2 xrobo_node`

Running the visualizer: `ros2 run xrobo_ros2 visualizer`

Running the body visualizer: `ros2 run xrobo_ros2 body_visualizer`

## Communication Architecture

This repo is designed for low-latency communication and ease-of-use. Generally, any additional nodes are only introduced if they are performance-critical. Any python nodes use `JAX` to jit-compile the code for faster performance while remaining in Python

### Python nodes

`g1_control/core/control_node`
- Control logic, including policy inference and safety filter.

`g1_control/core/mimic_node`
- Sends a trajectory for the controller to track.

`g1_control/core/standing_node`
- Sends a default standing position for the controller to track.

### C++ nodes

`g1_control/src/mocap_node`
- Communicates with the Optitrack system for ground-truth info about the pelvis and feet.

`g1_control/src/sim_node`
- Provides a MuJoCo simulation environment for testing the controller, mimicing the same topics as published by Unitree.

`g1_control/src/hardware_interface_node`
- Serves as a bridge between the Unitree hardware and the control node. This node reads the high-frequency state information, applies some simple filtering, and re-publishes it with a python-friendly format and a lower frequency. When commands are received, this node will also send it to the robot in the desired Unitree message format

### Launch files

While each node can be launched in their own terminals, it will be easier to use the provided launch files to make sure no critical nodes are forgotten.

`g1_control/launch/g1_hardware.launch.py`
- Runs the communication pipeline for working with the hardware

`g1_control/launch/g1_sim.launch.py`
- Runs the communication pipeline and launches the simulation to send/receive the same Unitree messages as seen on hardware.

`g1_control/launch/mocap.launch.py`
- Launches the mocap node and the VRPN/ROS2 interface

## Terminal setup

Some notes:
- The sim node does not need to be run when connected to the hardware
- There is a separate terminal setup script for hardware and sim. The only difference is in the cyclonedds connection interface.

### Simulation node
```
cd constrainedmimic/cm_ws
source scripts/setup_sim.sh
ros2 launch g1_control g1_sim.launch.py
```

### Control node
```
cd constrainedmimic/cm_ws
source scripts/setup_sim.sh # or setup_hardware.sh for hardware
python -m g1_control.core.control_node
```

### Trajectory / Mimic node
```
cd constrainedmimic/cm_ws
source scripts/setup_sim.sh # or setup_hardware.sh for hardware
python -m g1_control.core.mimic_node
```

### Standing node
```
cd constrainedmimic/cm_ws
source scripts/setup_sim.sh # or setup_hardware.sh for hardware
python -m g1_control.core.standing_node
```

### Mocap node
```
cd constrainedmimic/cm_ws
source scripts/setup_sim.sh # or setup_hardware.sh for hardware
ros2 launch g1_control mocap.launch.py
```

Mocap notes for SRC:
- Wifi: `SRC-humanoid_5G`
- In Motive:
    - IP address in streaming settings set to `192.168.110.119`
    - All rigid body assets for this project selected in assets pane
    - VRPN streaming enabled

Connecting to the robot notes:
- In Ubuntu network settings, create a `Unitree` profile with a `Manual` IPv4 method and address `192.168.123.99` and netmask `255.255.255.0`
- Connect to the robot via Ethernet and activate the `Unitree` network profile
- Note that the ethernet interface in `setup_hardware.sh` is configured for my laptop, this will need to be changed if using a different computer
- (optional) Can ssh to the robot with `ssh unitree@192.168.123.164`