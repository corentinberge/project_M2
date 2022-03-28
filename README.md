# Command for the Yaskawa HC10 collaborative robot

This project was made by a group of student from the "Université Paul Sabatier" in Toulouse, France.  
The purpose of this project is to make a force feedback control for the collaborative robot Yaskawa HC10.  
You will find identification and control code + the ros code to make it run on the simulators rViz and Gazebo of ROS.

--------

## Prerequisites

### __For Windows users with WSL2__

If you want to use Docker on Windows, we recommend using Windows 11 with WSL2 and Ubuntu18.04.  
If you still want to use Winows 10, make sure you're using WSL2 ([check which version of WSL you're using](https://linuxhint.com/check-wsl-version/)).

| **:warning: WARNING**                 |
|:--------------------------------------|
| Other Linux distributions won't work! |

You will need to install Docker and Nvidia CUDA container to run it. All the instructions are [here](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).  
If you are not using Windows 11, consider using a X-server like [VcXsrv](https://sourceforge.net/projects/vcxsrv/), or you won't have any GUI.

### __For Windows users without WSL2__

If you want to install it without WSL2, it is possible, however you won't be able to display any GUI.  
For this, you will need to install [Docker Desktop for Windows](https://docs.docker.com/desktop/windows/install/)

### __For Ubuntu18.04 users__

Install [Docker for Ubuntu 18.04](https://www.hostinger.com/tutorials/how-to-install-docker-on-ubuntu#How_to_Install_Docker_on_Ubuntu_1804)

--------

## Installation procedure

Clone this repository to your the folder of your choice. We recommend clone it via SSH if you want to be able to commit/push after. If you don't know how to do it, check [this link](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

Go into the directory of the ros version you want to install in the docker :
> noetic :
```bash
cd project_M2/Docker/noetic
```

> melodic :
```bash
cd project_M2/Docker/melodic
```

Then build the Docker with (this will take a few minutes):
```bash
bash launch.sh
```

| **:warning: WARNING**                 |
|:--------------------------------------|
| Following commands were made to run docker with `ros:noetic`. If you want to run docker with `ros:melodic`, do not forget to replace it in the commands |

After (few) minutes, you will be able to run the Docker.
> for Ubuntu (native or VM):
``` bash
docker run -v $HOME/project_M2/hc10_ros/:$(id -un)/catkin_ws/src/hc10_ros ros:noetic
```

> for WSL2 on Windows 11 (if you want a GUI):
``` bash
sudo docker run --rm -it --privileged --net=host --ipc=host \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-e DISPLAY=$DISPLAY \
-v $HOME/project_M2/hc10_ros:/home/$(id -un)/catkin_ws/src/hc10_ros \
-v $HOME/project_M2/dependencies_ros:/home/$(id -un)/catkin_ws/src/dependencies_ros \
-e XAUTHORITY=/home/$(id -un)/.Xauthority \
-e DOCKER_USER_NAME=$(id -un) \
-e DOCKER_USER_ID=$(id -u) \
-e DOCKER_USER_GROUP_NAME=$(id -gn) \
-e DOCKER_USER_GROUP_ID=$(id -g) \
-e ROS_IP=127.0.0.1 ros:noetic
```
Tip: Create a .bash file with this command :)

> for Windows without WSL2, run Docker Destop and run your container!

Once you are in the docker, run the folowing commands every time you run the Docker or you make a change.

```bash
cd ~/catkin_ws
catkin_make
```
