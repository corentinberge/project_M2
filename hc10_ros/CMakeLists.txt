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

add_message_files(
  FILES
  Num.msg
)

add_service_files(
  FILES
  AddTwoInts.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

include_directories(include ${catkin_INCLUDE_DIRS})

catkin_package(
  # INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs geometry_msgs nav_msgs tf gazebo_ros message_runtime
  DEPENDS GAZEBO
)

if(CATKIN_ENABLE_TESTING)
  find_package(roslaunch REQUIRED)
  roslaunch_add_file_check(launch)
endif()

install(DIRECTORY images meshes launch rviz urdf
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

add_executable(talker src/talker.cpp)
target_link_libraries(talker ${catkin_LIBRARIES})

add_executable(listener src/listener.cpp)
target_link_libraries(listener ${catkin_LIBRARIES})

add_executable(add_two_ints_server src/add_two_ints_server.cpp)
target_link_libraries(add_two_ints_server ${catkin_LIBRARIES})

add_executable(add_two_int_client src/add_two_int_client.cpp)
target_link_libraries(add_two_int_client ${catkin_LIBRARIES})