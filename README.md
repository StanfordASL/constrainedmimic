# Constrained Whole-Body Tracking for Humanoid Robots

Safe whole-body control of the Unitree G1

## Overview

(TODO) Add overview of the repo and the python/ROS2 components

## Prerequisites

### Python, Ubuntu, and ROS2 Versioning 

These instructions will assume we are using Python 3.12.3, Ubuntu 24.04, and ROS2 Jazzy. Other 3.12.x versions should work with Jazzy, but in general, I recommend running `python3` outside of any virtual environments to check what version your system python is, and match that exactly.

The code should also work for Ubuntu 22.04/ROS2 Humble/Python 3.10.x, but you'll need to tweak things to make this work. For starters, you'll need to (at least) replace any instances of `jazzy` with `humble`.

### Fetching the code

```
git clone https://github.com/StanfordASL/constrainedmimic
cd constrainedmimic
git submodule update --init --recursive
```

### Virtual environment

We use `uv` to manage the virtual environment. If you don't have `uv` already installed, run the following:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
# Optional, but recommended:
# echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
```

Then, in the top-level `constrainedmimic` directory,
```
uv venv --python 3.12.3 --system-site-packages
source .venv/bin/activate
cd cm_control
uv pip install -e .
```

### MuJoCo

For some ROS2 nodes, I use the C++ interface for MuJoCo. This requires building from source. First, navigate back to wherever you prefer to install MuJoCo and then run:
```
git clone https://github.com/google-deepmind/mujoco
cd mujoco
git checkout 3.4.0
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/mujoco -DCMAKE_BUILD_TYPE=Release
cmake --build .
sudo cmake --install .
```
Note that I used version `3.4.0` and decided to install to `/opt/mujoco` -- you can change this as needed. Check out the MuJoCo [building from source documentation](https://mujoco.readthedocs.io/en/latest/programming/#building-from-source) for more details.


### Pico VR

Download the release deb from [this link](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases) according to your ubuntu version (for instance, `XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb` for Ubuntu 24.04). Then, install with `sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb`

#### Usage

In a fresh terminal, run `adb devices` to see if the device is connected. Then, set up reverse port forwarding:
```
adb reverse tcp:63901 tcp:63901
```
And verify the port forward is active:
```
adb reverse --list
```
In the headset app, enter `127.0.0.1` as the IP address for the PC service, if it did not automatically detect it.

Then, open the XRoboToolkit app on the computer. It should say Status: CONNECTED in the Pico now. You should be able to close the app now on the computer, but it is not necessary.


### ROS2 setup

First, make sure that you have ROS2 installed. See the ROS2 Jazzy installation guide [here](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)

Install the required dependencies from `unitree_ros2` to communicate via cyclonedds
```
sudo apt install ros-jazzy-rmw-cyclonedds-cpp
sudo apt install ros-jazzy-rosidl-generator-dds-idl
sudo apt install libyaml-cpp-dev
```
And some additional dependencies for this project
```
sudo apt install ros-jazzy-pinocchio
sudo apt install libglfw3-dev
sudo apt install ros-jazzy-vrpn
```

We'll also need to pull in some dependencies for `unitree_ros2`. In a fresh shell session (without sourcing `/opt/ros/jazzy/setup.bash`), run the following:
```
cd constrainedmimic/cm_ws/src/third_party/unitree_ros2/cyclonedds_ws/src
git clone https://github.com/ros2/rmw_cyclonedds -b jazzy
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x 
```
Then, navigate back to the top-level workspace (`constrainedmimic/cm_ws`) and run a build:
```
./scripts/build_all.sh
```
Before running the nodes in any new terminals, be sure to source the setup script. If working in sim,
```
source scripts/setup_sim.sh
```
or if working with hardware,
```
source scripts/setup_hardware.sh
```


## Citation

```
@article{morton2026constrained,
  author={Morton, Daniel and Mohnot, Pranit and Pavone, Marco},
  title={Constrained Whole-Body Tracking for Humanoid Robots},
  journal={arXiv preprint arXiv:2606.00374},
  year={2026},
}
```

For citing `frax`, use the following:
```
@article{morton2026frax,
  author={Morton, Daniel and Pavone, Marco},
  title={frax: Fast Robot Kinematics and Dynamics in JAX},
  journal={arXiv preprint arXiv:2604.04310},
  year={2026},
  note={ICRA 2026 Workshop on Frontiers of Optimization for Robotics},
}
```

For citing `CBFpy` or `OSCBF`, use the following:
```
@inproceedings{morton2025oscbf,
  author={Morton, Daniel and Pavone, Marco},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions}, 
  year={2025},
  pages={187-194},
  doi={10.1109/IROS60139.2025.11246389}
}
```
