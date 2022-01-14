# Command for the Yaskawa HC10 collaborative robot

This project was made by a group of student from the "Université Paul Sabatier" in Toulouse, France. 

The purpose of this project is to make a force feedback control for the collaborative robot Yaskawa HC10.

You will find identification and control code + the ros code to make it run on the simulator rViz of ROS.

## Installation procedure

### For Windows users

If you want to use Docker on Windows, you'll need to run it on WSL2 with Ubuntu18.04 or newest. We also recommend using Windows 11.

| **:warning: WARNING**                 |
|:--------------------------------------|
| Other Linux distributions won't work! |

You will need to install Docker and Nvidia CUDA container to run it. All the instructions are [here](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

### For Ubuntu18.04 or newest users

Clone this repository to your home folder.

Build the Docker with (this will take a few minutes):
```bash
bash project_M2/Docker/launch.sh
```

After few minutes, you just need to run the Docker.
> for Ubuntu (native or VM):
``` bash
docker run -v $HOME/project_M2/hc10_ros/:$(id -un)/catkin_ws/src/hc10_ros ros:noetic
```

> for WSL2 on Windows 11:
``` bash
sudo docker run --rm -it --privileged --net=host --ipc=host \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-e DISPLAY=$DISPLAY \
-v $HOME/project_M2/hc10_ros:/home/$(id -un)/catkin_ws/src/hc10_ros \
-e XAUTHORITY=/home/$(id -un)/.Xauthority \
-e DOCKER_USER_NAME=$(id -un) \
-e DOCKER_USER_ID=$(id -u) \
-e DOCKER_USER_GROUP_NAME=$(id -gn) \
-e DOCKER_USER_GROUP_ID=$(id -g) \
-e ROS_IP=127.0.0.1 ros:noetic
```
Tip: Create a .bash file with this command :)

