# Load ros noetic from Docker Hub
FROM ros:noetic

#Make shell as Ubuntu
SHELL ["/bin/bash", "-c"]

#For non interactive shell
ENV DEBIAN_FRONTEND=noninteractive

# Update source repositories
RUN apt-get update

# Install dependencies for all the Docker
RUN apt-get install -y \
    apt-utils \
    lsb-release \
    python3.8 \
    gnupg2 \
    git \
    ssh \
    wget \
    curl \
    python3-pip \
    jupyter \
    libqt5gui5 \
    qt5-default \
    freeglut3 \
    ipython3 \
    python3-scipy \
    python3-numpy \
    python3-matplotlib \
    python3-pil.imagetk \
    python3-rospkg \
    software-properties-common
RUN pip3 install \
    Pillow \
    meshcat \
    qpsolvers \
    minidom-ext \
    rosdep

# Add robotpkg source repository to install pinocchio & noetic source repository to install ros-noetic
RUN echo "deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" | sudo tee /etc/apt/sources.list.d/robotpkg.list \
    && curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -

# Update repositories
RUN apt-get update

# Install desktop full noetic && pinocchio
RUN apt-get install -y ros-noetic-desktop-full \
    && source /opt/ros/noetic/setup.bash
RUN apt-get install -y \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-industrial-msgs \
    ros-noetic-industrial-robot-client \
    ros-noetic-rospy ros-noetic-roslaunch \
    ros-noetic-effort-controllers \
    python3-catkin-tools \
    robotpkg-py38-pinocchio \
    robotpkg-py38-crocoddyl \
    robotpkg-py38-qt5-gepetto-viewer-corba 

# Create new user and switch on it
RUN useradd -ms /bin/bash linux && echo "linux:linux" | chpasswd \
    && adduser linux sudo \
    && usermod -aG sudo linux \
    && echo "linux  ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/linux \
    && echo "linux ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER linux
WORKDIR /home/linux

# Configure environment for pynocchio
RUN echo "export PATH=/opt/openrobots/bin:$PATH" >> $HOME/.bashrc \
    && echo "export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH" >> $HOME/.bashrc \
    && echo "export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH" >> $HOME/.bashrc \
    && echo "export PYTHONPATH=/opt/openrobots/lib/python3.8/site-packages:$PYTHONPATH" >> $HOME/.bashrc \
    && echo "export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH" >> $HOME/.bashrc

# Create catkin_ws directory 
RUN mkdir -p $HOME/catkin_ws/src

# Initialize directory and rosdep
# RUN git clone -b kinetic-devel https://github.com/ros-industrial/motoman.git $HOME/catkin_ws/src/motoman
RUN cd $HOME/catkin_ws && rosdep update

# Build ROS packages
RUN cd $HOME/catkin_ws && source /opt/ros/noetic/setup.bash && sudo rosdep install -y --from-paths src/ --ignore-src --rosdistro noetic && catkin_make install

# Configure envirnoment for ROS
RUN echo "source $HOME/catkin_ws/devel/setup.bash" >> $HOME/.bashrc

# Add ROS dependencies
RUN source $HOME/catkin_ws/devel/setup.bash
