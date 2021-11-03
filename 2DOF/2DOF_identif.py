from numpy import linalg
from numpy.core.fromnumeric import shape
from numpy.lib.nanfunctions import _nanmedian_small
from pinocchio.visualize import GepettoVisualizer
from pinocchio.robot_wrapper import RobotWrapper
import matplotlib.pyplot as plt
import scipy.linalg as sp
import pinocchio as pin
import numpy as np
import os

# currentDir = os.getcwd()
# os.chdir('../')
workingDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
NV     = robot.nv                       # number of joint velocity ... ???
NJOINT = robot.model.njoints            # number of links
gv     = robot.viewer.gui

# Step 2 - generate inertial parameters for all links (excepted the base link)

names = []
for i in range(1, NJOINT):
    names += ['m'+str(i), 'mx'+str(i), 'my'+str(i), 'mz'+str(i), 'Ixx'+str(i), 'Ixy'+str(i), 'Iyy'+str(i), 'Izx'+str(i), 'Izy'+str(i), 'Izz'+str(i)]

phi = []
for i  in range (1, NJOINT):
    phi.extend(model.inertias[i].toDynamicParameters())

print('shape of phi:\t', np.array(phi).shape)

# Step 3 - Generate input and output - 100 samples

nbSamples = 100 # number of samples

# Generate 100 inputs
q   = np.random.rand(NQ, nbSamples) * np.pi - np.pi/2  # -pi/2 < q   < pi/2
dq  = np.random.rand(NQ, nbSamples) * 10               #     0 < dq  < 10
ddq = np.ones((NQ, nbSamples))                         #         ddq = 1

tau = []
for i in range(nbSamples):
    tau.extend(pin.rnea(model, data, q[:, i], dq[:, i], ddq[:, i]))
print('Shape of tau:\t', np.array(tau).shape)

# Step 4 - Create IDM with pinocchio

W = []  # Regression vector
for i in range (nbSamples):
    W.extend(pin.computeJointTorqueRegressor(model,data,q[:, i],dq[:, i],ddq[:, i]))
print('Shape of W:\t', np.array(W).shape)

# Step 5 - Remove non dynamic effect columns then remove zero value columns then remove the parameters related to zero value columns at the end we will have a matix W_modified et Phi_modified

threshold  = 0.000001
W_modified = np.array(W[:])
tmp        = []
for i in range(len(phi)):
    if (np.dot([W_modified[:, i]], np.transpose([W_modified[:, i]]))[0][0] <= threshold):
        tmp.append(i)
tmp.sort(reverse=True)

phi_modified   = phi[:]
names_modified = names[:]
for i in tmp:
    W_modified     = np.delete(W_modified, i, 1)
    phi_modified   = np.delete(phi_modified, i, 0)
    names_modified = np.delete(names_modified, i, 0)

# print('shape of W_m:\t', W_modified.shape)
# print('shape of phi_m:\t', np.array(phi_modified).shape)

# Step 6 - QR decomposition + pivoting

(Q, R, P) = sp.qr(W_modified, pivoting=True)

# print('shape of Q:\t', np.array(Q).shape)
# print('shape of R:\t', np.array(R).shape)
# print('shape of P:\t', np.array(P).shape)

# Step 7 - Calculate base parameters

tmp = 0
for i in range(len(R[0])):
    if R[i,i] > threshold :
        tmp = i

R1 = R[:tmp+1, :tmp+1]
R2 = R[:tmp+1, tmp+1:]

Q1 = Q[:, :tmp+1]

for i in (tmp+1, len(P)-1):
    names.pop(P[i])

# print('Shape of R1:\t', np.array(R1).shape)
# print('Shape of R2:\t', np.array(R2).shape)
# print('Shape of Q1:\t', np.array(Q1).shape)

beta = np.dot(np.linalg.inv(R1), R2)
# print('Shape of res:\t', beta.shape)

# beta = np.round(res, 6)
# print(res)

phi_modified = np.dot(np.linalg.inv(R1), np.dot(Q1.T, tau))
W_modified   = np.dot(Q1, R1)

# print('Shape of phi_m:\t', np.array(phi_modified).shape)
# print('Shape of W_m:\t', np.array(W_modified).shape)

inertialParameters = {names_modified[i] : phi_modified[i] for i in range(len(phi_modified))}
print(inertialParameters)

# print("press enter to continue")
# input()
gv.deleteNode('world', True)  # name, all=True
