# MRAC UR Perception
Author: [Huanyu Li](https://github.com/HuanyuL), DeepSeek

This repository contains the necessary tools and configurations for depth camera perception using UR10e robots. It supports the **ZED Camera** and **Azure Kinect**, along with MoveIt configurations, the Industrial Reconstruction package, and a Commander Node for easy manipulation of UR robots.

## Prerequisites

Before using this repository, ensure you have the following installed and configured:

1. **Docker**: Install Docker to containerize the environment.
   - [Install Docker](https://docs.docker.com/get-docker/)

2. **NVIDIA Container Toolkit**: Required for GPU acceleration with Docker.
   - [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

3. **Visual Studio Code (VSCode)**: Recommended for development and debugging.
   - [Download VSCode](https://code.visualstudio.com/download)

5. **UR10e Robot**: Ensure your UR10e robot is properly configured and connected. Please setup your machine ip address to **192.168.56.1**

## Repository Overview

The repository includes the following components:

1. **Camera Drivers**:
   - ZED Camera driver for depth perception.
   - Azure Kinect driver for depth perception.

2. **MoveIt Configurations**:
   - Pre-configured MoveIt setup for UR10e robots with the end-effector.

3. **Industrial Reconstruction Package**:
   - Tools for 3D reconstruction and environment mapping.

4. **Commander Node**:
   - A user-friendly node for easy manipulation and control of UR robots.

## Setup Instructions

### 1. Clone the Repository
Clone this repository to your local machine, remeber fork it if you want to develop your application:
```bash
git clone https://github.com/your-username/MRAC-UR-Perception.git
cd MRAC-UR-Perception
```
### 2. Build the Docker Image
To build the image.
```
.docker/build_image.sh
```
To run the image.
```
.docker/run_user.sh
```
You may need to change the owner of the dev_ws, copy the line showing on the terminal.
```
sudo chown -R [YOUR USER NAME] /dev_ws
```
Start a terminal
```
terminator
```
### 3. Launch the Perception Package
Inside the Docker container, launch the desired perception pipeline:

For ZED Camera:

```bash
roslaunch zed_wrapper zedm.launch
```
For Azure Kinect:
```bash
roslaunch azure_kinect_ros_driver driver.launch
```
### 4. Launch the robot driver
If you want to simulate the robot with a fake controller:
```bash
roslaunch ur10e_moveit_config demo.launch
```
If you want to connect to the real robot with zed camera mount on the end-effector:
```bash
roslaunch commander ur10e_zed_commander.launch
```
If you want to connect the real robot with the Azure Kinect stationary placed:
```bash
roslaunch commander ur10e_ka_commander.launch
```
### 5. Hand-eye calibration
The tutorial for hand-eye application can be sourced from this [link](https://github.com/moveit/moveit_tutorials/blob/master/doc/hand_eye_calibration/hand_eye_calibration_tutorial.rst)

1. Choose the camera image topics from your camera
2. Choose the frames according to your application
3. FreeDrive the robot to different poses, click save sample, and repeat this step until you receive 10-15 samples
4. Save the camera pose to a new launch file
5. Integrate the camera pose to your project's launch node

### 6. Usage Examples
Example 1: 3D Reconstruction
Launch the Industrial Reconstruction package:

```bash
roslaunch commander reconstruction.launch
```
Use the Commander nodebook to move the robot and capture data.


## Troubleshooting
Docker GPU Issues: Ensure the NVIDIA Container Toolkit is installed and configured correctly.

Camera Connectivity: Check the camera connections and ensure the drivers are properly installed.

ROS Communication: Verify that the ROS master is running and all nodes are communicating.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.