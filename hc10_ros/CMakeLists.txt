cmake_minimum_required(VERSION 3.0.2)
project(hc10_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf
  gazebo_ros
  message_generation
)

find_package(gazebo REQUIRED)

# add_message_files(
#   FILES
#   JointCommands.msg
#   JointState.msg
#   JointState_noHeader.msg
# )

# add_service_files(
#   FILES
#   AddTwoInts.srv
#   joint_state.srv
# )

# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

include_directories(include ${catkin_INCLUDE_DIRS})

catkin_package(
  # INCLUDE_DIRS include
  CATKIN_DEPENDS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf
  gazebo_ros
  message_runtime
  DEPENDS GAZEBO
)

if(CATKIN_ENABLE_TESTING)
  find_package(roslaunch REQUIRED)
  roslaunch_add_file_check(launch)
endif()

install(
  DIRECTORY
  images
  meshes
  launch
  rviz
  urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_install_python(
  PROGRAMS
  scripts/talker.py
  #scripts/test.py
  scripts/execute_trajectory.py
  #scripts/traj.py
  #scripts/test_traj.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
