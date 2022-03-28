# How to launch the proejct in ROS

### __Simulation :__

If you want to simulate in Gazebo environment, you just need to type this command line :
```bash
roslaunch hc10_ros simulation_gazebo.launch
```
This command will launch just the robot in an empty world. To launch the robot in the real environment, you need to type this command line :
```bash
roslaunch hc10_ros simulation_gazebo_with_env.launch
```

### __On robot__ :

If you want to send command to the robot with MoveIt, a plugin on RVIZ, you just need to type this command line :
```bash
roslaunch hc10_ros simulation_moveit_rviz.launch sim:=false robot_ip:=192.168.1.40 controller:=yrc1000
```

# Usefull topics

### **/motoman_hc10/joint{$}_position_controller/command**

topic controlling the pose of the joint {$} in radiant.

*/!\ All joints range between [-3.141592653589793;+3.141592653589793], except joint 3 for which the pose control ranges between [0;6.28318530718] !!!*

### **/motoman_hc10/joint_states/**

topic retrieving the effort, position, velocity and name of the joints.
to access a specific joint data, do: 
```
rostopic echo /motoman_hc10/joint_states/effort[{$}]
rostopic echo /motoman_hc10/joint_states/position[{$}]
rostopic echo /motoman_hc10/joint_states/velocity[{$}]
```

where {$} is the number of the joint minus 1 (the list always starts at 0).

```
rostopic echo /motoman_hc10/joint_states/name
```
returns the current names of the joints being published.
