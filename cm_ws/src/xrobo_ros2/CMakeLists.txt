cmake_minimum_required(VERSION 3.8)
project(xrobo_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rosidl_default_generators REQUIRED)

include_directories(/opt/apps/roboticsservice/SDK/x64)
link_directories(/opt/apps/roboticsservice/SDK/x64)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/XRoboState.msg"
  DEPENDENCIES geometry_msgs
)

add_executable(xrobo_node src/xrobo_node.cpp)
target_include_directories(xrobo_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(xrobo_node
  PXREARobotSDK
  pthread
  Eigen3::Eigen)
ament_target_dependencies(xrobo_node
  rclcpp
  geometry_msgs
)
rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")
target_link_libraries(xrobo_node "${cpp_typesupport_target}")

install(TARGETS 
  xrobo_node
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  scripts/frame_visualizer.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME frame_visualizer)

install(PROGRAMS
  scripts/body_visualizer.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME body_visualizer)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cppcheck_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
