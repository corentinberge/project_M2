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