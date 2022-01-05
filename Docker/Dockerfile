# Load ros noetic from Docker Hub
FROM ros:noetic

#Make shell as Ubuntu
SHELL ["/bin/bash", "-c"]

# Update source repositories
RUN apt-get update

# Install dependencies for all the Docker
RUN apt-get install -qqy lsb-release gnupg2 curl git ipython3 python3-pip freeglut3 python3-matplotlib python3-pil.imagetk jupyter python3-scipy python3-matplotlib jupyter python3-scipy python3-numpy libqt5gui5 qt5-default wget software-properties-common ssh
RUN pip3 install meshcat qpsolvers

# Install vscode
RUN wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | apt-key add - && add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
RUN apt-get install -y code

# Add robotpkd source repository to install pinocchio
RUN echo "deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" | sudo tee /etc/apt/sources.list.d/robotpkg.list
RUN curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -

# Update repositories
RUN apt-get update

# Install desktop full noetic && pinocchio
RUN apt-get install -y ros-noetic-desktop-full robotpkg-py38-pinocchio robotpkg-py38-crocoddyl robotpkg-py38-qt5-gepetto-viewer-corba

# Create new user and switch on it
RUN useradd -ms /bin/bash linux && echo "linux:linux" | chpasswd && adduser linux sudo
RUN usermod -aG sudo linux
RUN echo "linux  ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/linux
RUN echo "linux ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER linux
WORKDIR /home/linux

# Configure environment for pynocchio
RUN echo "export PATH=/opt/openrobots/bin:$PATH" >> $HOME/.bashrc
Run echo "export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH" >> $HOME/.bashrc
RUN echo "export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH" >> $HOME/.bashrc
RUN echo "export PYTHONPATH=/opt/openrobots/lib/python3.8/site-packages:$PYTHONPATH" >> $HOME/.bashrc
RUN echo "export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH" >> $HOME/.bashrc

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
