# Load ros noetic from Docker Hub
FROM ros:noetic

#Make shell as Ubuntu
SHELL ["/bin/bash", "-c"]

#For non interactive shell
ENV DEBIAN_FRONTEND=noninteractive

# Update source repositories
RUN apt-get update \
    # Install dependencies for all the Docker
    && apt-get install -y \
    python3.8 \
    apt-utils \
    curl \
    wget \
    python3-pip \
    libqt5gui5 \
    qt5-default \
    ipython3 \
    python3-matplotlib \
    python3-scipy \
    python3-numpy \
    python3-rospkg \
    python3-rosdep \
    # Install Python packages
    && pip3 install \
    Pillow \
    meshcat \
    qpsolvers \
    minidom-ext \
    # Add robotpkg source repository to install pinocchio & noetic source repository to install ros-noetic
    && echo "deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" | sudo tee /etc/apt/sources.list.d/robotpkg.list \
    && curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add - \
    # Update repositories
    && apt-get update \
    # Install desktop full noetic && pinocchio
    && apt-get install -y ros-noetic-desktop \
    && source /opt/ros/noetic/setup.bash \
    # Install supplementary required packages
    # Install supplementary required packages
    && apt-get install -y \
    ros-noetic-rospy \
    ros-noetic-roslaunch \
    ros-noetic-gazebo-ros \
    ros-noetic-gazebo-ros-control \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-industrial-msgs \
    ros-noetic-industrial-robot-client \
    ros-noetic-effort-controllers \
    ros-noetic-industrial-core \
    robotpkg-py38-pinocchio \
    robotpkg-py38-crocoddyl \
    robotpkg-py38-qt5-gepetto-viewer-corba \
    python3-osrf-pycommon \
    python3-catkin-pkg \
    python3-catkin-tools \
    # Create new user
    && useradd -ms /bin/bash linux \
    && echo "linux:linux" | chpasswd \
    && adduser linux sudo \
    && usermod -aG sudo linux \
    && echo "linux  ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/linux \
    && echo "linux ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
# Switch on this new user
USER linux
WORKDIR /home/linux

# Configure environment for pynocchio
RUN echo "export PATH=/opt/openrobots/bin:$PATH" >> $HOME/.bashrc \
    && echo "export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH" >> $HOME/.bashrc \
    && echo "export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH" >> $HOME/.bashrc \
    && echo "export PYTHONPATH=/opt/openrobots/lib/python3.8/site-packages:$PYTHONPATH" >> $HOME/.bashrc \
    && echo "export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH" >> $HOME/.bashrc \
    # Create catkin_ws directory 
    && mkdir -p $HOME/catkin_ws/src \
    # Initialize directory and rosdep
    && cd $HOME/catkin_ws \
    && rosdep update \
    # Build ROS packages
    && cd $HOME/catkin_ws \
    && source /opt/ros/noetic/setup.bash \
    && sudo rosdep install -y --from-paths src/ --ignore-src --rosdistro noetic \
    && catkin_make install \
    # Configure envirnoment for ROS
    && echo "source $HOME/catkin_ws/devel/setup.bash" >> $HOME/.bashrc \
    # Add ROS dependencies
    && source $HOME/catkin_ws/devel/setup.bash \
    # Clean apt cache
    && sudo apt-get autoclean \
    && sudo apt-get clean \
    && sudo apt-get autoremove