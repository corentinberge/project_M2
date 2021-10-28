from numpy.lib.nanfunctions import _nanmedian_small
from pinocchio.visualize import GepettoVisualizer
from pinocchio.robot_wrapper import RobotWrapper
import matplotlib.pyplot as plt
import pinocchio as pin
import numpy as np
import math
import os


currentDir = os.getcwd()
os.chdir('../')
workingDir = os.getcwd()

# urdf directory path
package_path = workingDir
urdf_path = package_path + '/robots/urdf/planar_2DOF.urdf'
robot = RobotWrapper()

#robot.BuildFromURDF(urdf,path,verbose=True)
robot.initFromURDF(urdf_path,package_path,verbose=True)
#print("MODEL DU ROBOT\n",robot.model)
#print("VISUAL MODEL ",robot.visual_model)

robot.initViewer(loadModel=True)
robot.display(robot.q0)
# loading robot datas (data,model,NQ,NV,Njoint)
data = robot.data
model = robot.model
NQ = robot.nq #number of joint angle 
NV = robot.nv #number of joint velocity 
NJOINT = robot.model.njoints
gv = robot.viewer.gui

names = []
for i in [1 ,2]:
    print(i)
    names += ['m'+str(i), 'mx'+str(i), 'my'+str(i), 'mz'+str(i), 'Ixx'+str(i), 'Ixy'+str(i), 'Iyy'+str(i), 'Izx'+str(i), 'Izy'+str(i), 'Izz'+str(i)]

print("tout les noms : ",names,"taille des noms ",len(names))
dict = {}
for i in range(len(names)):
    print(i)
    dict[names[i]] = 0

print("dictionnaire \n",dict)
print("taille dictionnaire :",len(dict))

print(NQ)
samples = 100 # number of samples
q = np.random.rand(NQ,samples)*(np.pi) - np.pi/2# -pi<q<2pi
dq = np.random.rand(NQ,samples)*10
ddq = np.ones((NQ,samples))

tau = np.zeros((NQ,samples))
#print(tau)
for i in range(samples):
    tau[:,i] = np.transpose(pin.rnea(model,data,q[:,i],ddq[:,i],ddq[:,i])) #calcul du tau réel 
print("size : ",tau.size)




print("press enter to continue")
input()
gv.deleteNode('world', True)  # name, all=True
#on fais bouger le robot pour le test 
