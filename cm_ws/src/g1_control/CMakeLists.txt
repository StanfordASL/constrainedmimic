cmake_minimum_required(VERSION 3.8)
project(g1_control)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(unitree_hg REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(g1_control_msgs REQUIRED)
find_package(xrobo_ros2 REQUIRED)
find_package(Eigen3 REQUIRED)

# MuJoCo configuration
set(MUJOCO_DIR "/opt/mujoco")
include_directories(${MUJOCO_DIR}/include)
link_directories(${MUJOCO_DIR}/lib)

find_package(OpenGL REQUIRED)
find_package(glfw3 QUIET)
if(NOT glfw3_FOUND)
  set(GLFW_LIBS "glfw")
else()
  set(GLFW_LIBS glfw)
endif()

# Install Python package code
ament_python_install_package(${PROJECT_NAME})

# C++ Executables
add_executable(sim_node src/sim_node.cpp)
add_executable(mocap_node src/mocap_node.cpp)
set(UNITREE_EXAMPLE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/unitree_ros2/example/src)
add_executable(hardware_interface_node
  src/hardware_interface_node.cpp
  ${UNITREE_EXAMPLE_SRC}/src/common/motor_crc_hg.cpp
)
target_include_directories(hardware_interface_node PRIVATE
  ${UNITREE_EXAMPLE_SRC}/include
  ${UNITREE_EXAMPLE_SRC}/include/common
)
add_executable(teleop_visualizer_node src/teleop_visualizer_node.cpp)
add_executable(mimic_visualizer_node src/mimic_visualizer_node.cpp)

# Target dependencies and linking
ament_target_dependencies(sim_node
  rclcpp
  unitree_hg
  sensor_msgs
  std_msgs
  nav_msgs
  geometry_msgs
  g1_control_msgs
)
target_link_libraries(sim_node mujoco ${OPENGL_LIBRARIES} ${GLFW_LIBS})

ament_target_dependencies(teleop_visualizer_node rclcpp g1_control_msgs)
target_link_libraries(teleop_visualizer_node mujoco ${OPENGL_LIBRARIES} ${GLFW_LIBS})

ament_target_dependencies(mimic_visualizer_node rclcpp std_msgs g1_control_msgs)
target_link_libraries(mimic_visualizer_node mujoco ${OPENGL_LIBRARIES} ${GLFW_LIBS} Eigen3::Eigen)

ament_target_dependencies(mocap_node rclcpp sensor_msgs std_msgs geometry_msgs g1_control_msgs)
target_link_libraries(mocap_node Eigen3::Eigen)

ament_target_dependencies(hardware_interface_node rclcpp unitree_hg g1_control_msgs diagnostic_msgs std_msgs)
target_link_libraries(hardware_interface_node Eigen3::Eigen)

# Install C++ targets
install(TARGETS sim_node mocap_node hardware_interface_node teleop_visualizer_node mimic_visualizer_node
  DESTINATION lib/${PROJECT_NAME})

# Add RPATH for MuJoCo
set_target_properties(sim_node teleop_visualizer_node mimic_visualizer_node PROPERTIES
  INSTALL_RPATH "${MUJOCO_DIR}/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)

# Install Python nodes from scripts directory
install(PROGRAMS
  scripts/hardware_debugging_node.py
  scripts/mimic_node.py
  scripts/control_node.py
  scripts/control_node_teleop.py
  scripts/policy_node.py
  scripts/standing_node.py
  scripts/teleop_node.py
  scripts/karate_chop_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
