# Load ros melodic from Docker Hub
FROM ros:melodic

#Make shell as Ubuntu
SHELL ["/bin/bash", "-c"]

# Update source repositories
RUN apt-get update

# Install dependencies for all the Docker
RUN apt-get install -y \
    apt-utils \
    # lsb-release \
    # python \
    # gnupg2 \
    curl \
    # git \
    # ssh \
    wget \
    python-pip \
    # jupyter \
    # freeglut3 \
    libqt5gui5 \
    qt5-default 
# software-properties-common
RUN pip install \
    Pillow \
    meshcat \
    qpsolvers==1.4.1 \
    minidom-ext \
    ipython \
    scipy \
    numpy \
    matplotlib \
    # pil.imagetk \
    rospkg \
    rosdep

# Install vscode
# RUN wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | apt-key add - \
#     && add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
# RUN apt-get install -y code

# Add robotpkg source repository to install pinocchio & melodic source repository to install ros-melodic
RUN echo "deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" | sudo tee /etc/apt/sources.list.d/robotpkg.list
RUN curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -

# Update repositories
RUN apt-get update

# Install desktop full melodic && pinocchio
RUN apt-get install -y ros-melodic-desktop \
    && source /opt/ros/melodic/setup.bash
RUN apt-get install -y \
    # ros-melodic-moveit \
    # ros-melodic-moveit-visual-tools \
    ros-melodic-rospy \
    ros-melodic-roslaunch \
    ros-melodic-gazebo-ros \
    ros-melodic-gazebo-ros-control \
    ros-melodic-ros-control \
    ros-melodic-ros-controllers \
    ros-melodic-industrial-msgs \
    ros-melodic-industrial-robot-client \
    ros-melodic-effort-controllers \
    python-catkin-tools \
    robotpkg-py27-pinocchio \
    robotpkg-py27-crocoddyl \
    robotpkg-py27-qt5-gepetto-viewer-corba 

# Create new user and switch on it
RUN useradd -ms /bin/bash linux \
    && echo "linux:linux" | chpasswd \
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
    && echo "export PYTHONPATH=/opt/openrobots/lib/python2.7/site-packages:$PYTHONPATH" >> $HOME/.bashrc \
    && echo "export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH" >> $HOME/.bashrc

# Create catkin_ws directory 
RUN mkdir -p $HOME/catkin_ws/src

# Initialize directory and rosdep
# RUN git clone -b kinetic-devel https://github.com/ros-industrial/motoman.git $HOME/catkin_ws/src/motoman
RUN cd $HOME/catkin_ws \
    && rosdep update

# Build ROS packages
RUN cd $HOME/catkin_ws \
    && source /opt/ros/melodic/setup.bash \
    && sudo rosdep install -y --from-paths src/ --ignore-src --rosdistro melodic \
    && catkin_make install

# Configure envirnoment for ROS
RUN echo "source $HOME/catkin_ws/devel/setup.bash" >> $HOME/.bashrc

# Add ROS dependencies
RUN source $HOME/catkin_ws/devel/setup.bash
