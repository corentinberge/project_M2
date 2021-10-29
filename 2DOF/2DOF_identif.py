from numpy.core.fromnumeric import shape
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

inertialParameters = {names[i] : phi[i] for i in range(len(names))}

# Step 3 - Generate input and output - 100 samples

nbSamples = 100 # number of samples

# Generate 100 inputs
q   = np.random.rand(NQ, nbSamples) * np.pi - np.pi/2  # -pi/2 < q   < pi/2
dq  = np.random.rand(NQ, nbSamples) * 10               #     0 < dq  < 10
ddq = np.ones((NQ, nbSamples))                         #         ddq = 1

# Generate 100 outputs --- WIP (A voir avec le doctorant)
# tau = np.array([np.transpose(pin.rnea(model, data, q[:, i], ddq[:, i], ddq[:, i])) for i in range(nbSamples)])
# print(tau.shape)

tau = []
for i in range(nbSamples):
    tau.extend(pin.rnea(model, data, q[:, i], ddq[:, i], ddq[:, i]))
print('Shape of tau:\t', np.array(tau).shape)

# Step 4 - Create IDM with pinocchio

# W = np.array([pin.computeJointTorqueRegressor(model,data,q[:, i],dq[:, i],ddq[:, i]) for i in range (nbSamples)])
# print(W.shape)

W = []
for i in range (nbSamples):
    W.extend(pin.computeJointTorqueRegressor(model,data,q[:, i],dq[:, i],ddq[:, i]))
print('Shape of W:\t', np.array(W).shape)


# Step 5 - Remove non dynamic effect columns then remove zero value columns then remove the parameters related to zero value columns
#           at the end we will have a matix W_modifier et Phi_modifier 
# the parameters are not calculated so the modification will be or in the name of the parameter or in the size of the parameter vector we have to check 
# the size of the first vector calculated in previous and comparet with the size of W and tau 
# best regards
# Julien est un idiot avec ca methode agile 


threshold = 0.000001

# test = [[1, 2, 3, 4, 5]]
# print(test)
# print(np.transpose(test))
# print()
# print(np.dot(test, np.transpose(test)))

# test = [[1, 2, 0, 4, 5],
#         [6, 7, 0, 9, 10],
#         [11, 12, 0, 14, 15],
#         [16, 17, 0, 19, 20]]

W_base = W[:]
W = np.array(W)
tmp = []
for i in range(20):
    # print(np.dot([W[:, i]], np.transpose([W[:, i]])))
    if (np.dot([W[:, i]], np.transpose([W[:, i]]))[0][0] <= threshold):
        #W = np.delete(W, i, 1)
        tmp.append(i)
tmp.sort(reverse=True)
print(tmp)

phi_base = phi[:]
# inertialParameters_base = inertialParameters[:]

for i in tmp:
    W = np.delete(W, i, 1)
    phi = np.delete(phi, i, 0)

print('shape of Wb:\t', np.array(W_base).shape)
print('shape of W:\t', W.shape)

print('shape of phi:\t', np.array(phi).shape)



# j=0
# W_base = W[:]
# for i in range(100,200):
#     # print(np.transpose([W[i]]))
#     # print([W[i]])
#     print(np.dot([W[i]], np.transpose([W[i]])))
        





# print("press enter to continue")
# input()
gv.deleteNode('world', True)  # name, all=True
