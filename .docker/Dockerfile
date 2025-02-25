ARG ROS_DISTRO=noetic

FROM ros:$ROS_DISTRO-ros-base AS base


ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm-256colorZ

FROM base AS ros

RUN apt-get update && apt-get install -y --no-install-recommends\
    ssh \
    git \
    curl \
    wget \
    lsb-release \
    terminator \
    build-essential \
    cmake \
    python3-pip \
    python3-flake8 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


RUN pip3 install --no-cache-dir --upgrade pip\
    argcomplete \
    flake8-blind-except \
    flake8-builtins \
    flake8-comprehensions \
    flake8-deprecated \
    flake8-return \
    flake8-length \
    flake8-todo \
    flake8-quotes \
    black \
    mypy \
    prospector \
    pylint \
    numpy \
    ipykernel \
    opencv-python \
    pyquaternion \
    python-dateutil \
    matplotlib \
    cython \
    pytransform3d \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends\
    pkg-config \
    python3-catkin-tools \
    python3-rosdep \
    python3-rosinstall-generator \
    python3-vcstool \
    python3-pykdl \
    ros-$ROS_DISTRO-rqt \
    ros-$ROS_DISTRO-rqt-common-plugins \
    ros-$ROS_DISTRO-rqt-robot-plugins \
    ros-$ROS_DISTRO-rqt-joint-trajectory-controller \   
    ros-$ROS_DISTRO-roslint \
    ros-$ROS_DISTRO-rqt-gui \
    ros-$ROS_DISTRO-rqt-gui-py \
    ros-$ROS_DISTRO-rqt-py-common \
    ros-$ROS_DISTRO-moveit \
    ros-$ROS_DISTRO-moveit-visual-tools \
    ros-$ROS_DISTRO-handeye \
    ros-$ROS_DISTRO-baldor \
    ros-$ROS_DISTRO-criutils \
    ros-$ROS_DISTRO-trac-ik \
    ros-$ROS_DISTRO-kdl-parser \
    ros-$ROS_DISTRO-urdf-parser-plugin \
    ros-$ROS_DISTRO-urdfdom-py \
    ros-$ROS_DISTRO-kdl-parser-py \
    ros-$ROS_DISTRO-ros-industrial-cmake-boilerplate \
    ros-$ROS_DISTRO-industrial-robot-client \
    ros-$ROS_DISTRO-plotjuggler-ros \
    ros-$ROS_DISTRO-ur-client-library \
    ros-$ROS_DISTRO-ur-msgs \
    ros-$ROS_DISTRO-universal-robots \
    ros-$ROS_DISTRO-controller-manager \
    ros-$ROS_DISTRO-pass-through-controllers \
    ros-$ROS_DISTRO-force-torque-sensor-controller \
    ros-$ROS_DISTRO-industrial-robot-status-interface \
    ros-$ROS_DISTRO-industrial-robot-status-controller \
    ros-$ROS_DISTRO-joint-state-controller \
    ros-$ROS_DISTRO-joint-state-publisher \
    ros-$ROS_DISTRO-joint-trajectory-controller \
    ros-$ROS_DISTRO-cartesian-trajectory-controller \
    ros-$ROS_DISTRO-scaled-joint-trajectory-controller \
    ros-$ROS_DISTRO-speed-scaling-interface \
    ros-$ROS_DISTRO-speed-scaling-state-controller \
    ros-$ROS_DISTRO-velocity-controllers \
    ros-$ROS_DISTRO-effort-controllers \
    ros-$ROS_DISTRO-diagnostic-updater \
    ros-$ROS_DISTRO-image-geometry \
    ros-$ROS_DISTRO-image-transport-plugins \
    ros-$ROS_DISTRO-camera-info-manager \
    ros-$ROS_DISTRO-xacro \
    ros-$ROS_DISTRO-rgbd-launch \
    ros-$ROS_DISTRO-robot-state-publisher \
    ros-$ROS_DISTRO-rtabmap \
    ros-$ROS_DISTRO-rtabmap-ros \
    ros-$ROS_DISTRO-depthimage-to-laserscan \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


WORKDIR /
RUN mkdir -p dev_ws/src
WORKDIR /dev_ws/src

# niceness permissions (for ur driver use without Prempt-RT)
RUN echo "* - nice -15" >> /etc/security/limits.conf

FROM ros AS ka

RUN curl -sSL https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4/libk4a1.4_1.4.1_amd64.deb > /tmp/libk4a1.4_1.4.1_amd64.deb
RUN curl -sSL https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4-dev/libk4a1.4-dev_1.4.1_amd64.deb > /tmp/libk4a1.4-dev_1.4.1_amd64.deb
RUN curl -sSL https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4abt1.1/libk4abt1.1_1.1.2_amd64.deb > /tmp/libk4abt1.1_1.1.2_amd64.deb
RUN curl -sSL https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4abt1.1-dev/libk4abt1.1-dev_1.1.2_amd64.deb > /tmp/libk4abt1.1-dev_1.1.2_amd64.deb
RUN curl -sSL https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/k/k4a-tools/k4a-tools_1.4.1_amd64.deb > /tmp/k4a-tools_1.4.1_amd64.deb
RUN echo 'libk4a1.4 libk4a1.4/accepted-eula-hash string 0f5d5c5de396e4fee4c0753a21fee0c1ed726cf0316204edda484f08cb266d76' | debconf-set-selections
RUN echo 'libk4abt1.1	libk4abt1.1/accepted-eula-hash	string	03a13b63730639eeb6626d24fd45cf25131ee8e8e0df3f1b63f552269b176e38' | debconf-set-selections
RUN dpkg -i /tmp/libk4a1.4_1.4.1_amd64.deb
RUN dpkg -i /tmp/libk4a1.4-dev_1.4.1_amd64.deb
RUN dpkg -i /tmp/libk4abt1.1_1.1.2_amd64.deb
RUN dpkg -i /tmp/libk4abt1.1-dev_1.1.2_amd64.deb
RUN apt-get update
RUN apt-get install -y libsoundio1
RUN dpkg -i /tmp/k4a-tools_1.4.1_amd64.deb
# fix k4abtConfig.cmake Error: Azure Kinect SDK Version numbers contain exactly 3 components
RUN sed -i 's/find_dependency(k4a 1.4 REQUIRED)/find_dependency(k4a 1.4.1 REQUIRED)/' /usr/lib/cmake/k4abt/k4abtConfig.cmake

RUN pip3 install --no-cache-dir --upgrade pyk4a

FROM ka AS open3d

RUN pip3 install --no-cache-dir --upgrade open3d --ignore-installed PyYAML

FROM open3d AS build

COPY . /dev_ws/src/

RUN pip install --force-reinstall --no-deps numpy==1.21.6

WORKDIR /dev_ws

# RUN chown -R sashanosense /usr/local/lib/python3.8/dist-packages/

RUN ["/bin/bash", "-c", "source /opt/ros/$ROS_DISTRO/setup.bash &&\
    catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release &&\
    catkin build \
    "]


FROM build AS bash

RUN ["/bin/bash", "-c", "cp /dev_ws/src/.docker/setup.bash /dev_ws/setup.bash && chmod 777 /dev_ws/setup.bash"]
RUN ["/bin/bash", "-c", "cp /dev_ws/src/.docker/entrypoint.bash / && chmod 777 /entrypoint.bash"]
ENTRYPOINT ["bash", "/entrypoint.bash" ]

CMD ["bash"]


FROM build as dev

SHELL ["zsh", "chsh -s ~/.zshrc" ]

RUN ["/bin/bash", "-c", "cp /dev_ws/src/.docker/setup.zsh /dev_ws/setup.zsh && chmod 777 /dev_ws/setup.zsh"]
RUN ["/bin/bash", "-c", "cp /dev_ws/src/.docker/entrypoint.zsh / && chmod 777 /entrypoint.zsh"]

ENTRYPOINT ["zsh", "/entrypoint.zsh" ]

CMD ["zsh"]