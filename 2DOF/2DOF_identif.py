from numpy.lib.nanfunctions import _nanmedian_small
from pinocchio.visualize import GepettoVisualizer
from pinocchio.robot_wrapper import RobotWrapper
import matplotlib.pyplot as plt
import pinocchio as pin
import numpy as np
import os

currentDir = os.getcwd()
os.chdir('../')
workingDir = os.getcwd()

# urdf directory path
package_path = workingDir
urdf_path    = package_path + '/robots/urdf/planar_2DOF.urdf'

# Step 1 - load model, create robot model and create robot data

robot = RobotWrapper()
robot.initFromURDF(urdf_path, package_path,verbose=True)
robot.initViewer(loadModel=True)
robot.display(robot.q0)

data   = robot.data
model  = robot.model
NQ     = robot.nq                       # number of joint angle 
NV     = robot.nv                       # number of joint velocity 
NJOINT = robot.model.njoints            # number of links
gv     = robot.viewer.gui

# Step 2 - generate inertial parameters for all joint 
names = []
for i in range(NJOINT):
    names += ['m'+str(i), 'mx'+str(i), 'my'+str(i), 'mz'+str(i), 'Ixx'+str(i), 'Ixy'+str(i), 'Iyy'+str(i), 'Izx'+str(i), 'Izy'+str(i), 'Izz'+str(i)]

values = []
for inertia in model.inertias:
    values.extend(inertia.toDynamicParameters())

print(len(values))

inertialParameters = {names[i] : values[i] for i in range(len(names))}

# Step 3 - Generate input and output - 100 samples

nbSamples = 100 # number of samples

# Generate 100 inputs
q   = np.random.rand(NQ, nbSamples) * np.pi - np.pi/2  # -pi/2 < q   < pi/2
dq  = np.random.rand(NQ, nbSamples) * 10               #     0 < dq  < 10
ddq = np.ones((NQ, nbSamples))                         #         ddq = 1

# Generate 100 outputs --- WIP (A voir avec le doctorant)
tau = np.array([np.transpose(pin.rnea(model, data, q[:, i], ddq[:, i], ddq[:, i])) for i in range(nbSamples)])
print(tau.shape)

# Step 4 - Create IDM with pinocchio



# print("press enter to continue")
# input()
gv.deleteNode('world', True)  # name, all=True
