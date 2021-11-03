from numpy.core.defchararray import join
from numpy.core.fromnumeric import transpose
import pinocchio as pin
from pinocchio.deprecated import forwardDynamics
from pinocchio.utils import *
from pinocchio.visualize import GepettoVisualizer
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import norm, inv, pinv
import time
import os

#Simulation de la trajectoire
def simulateurTraj(N):
    dt = 1e-2
    a0,a1,a2 = 0,1,2
    X = np.zeros((N,7))
    for i in range(N-1):
        q,dq = loiPoly(a0,a1,a2,i*dt)
        q = np.array(q)
        loi(q,X)
        #robot.forwardKinematics(q) #update joint 
        pin.updateFramePlacements(robot.model,robot.data) #update frame placement  
        X[i,:] = pin.SE3ToXYZQUAT(robot.data.oMf[ID_F])
        robot.display(q)
        time.sleep(dt)
    print(X.shape)
        
#Loi polynomial
def loiPoly(a0,a1,a2,t):
    q = [a0 + a1*t + a2*(t**2) , a0 + a1*t + a2*(t**2)]
    dq = [ a1 + 2*a2*t , a1 + 2*a2*t ]
    return q, dq
   
#Jacobienne
def jacobienneValidation(J,Xp): 
    print()

def calculDotX(X):
    print("test")
    
#Loi de commande du système
def loi(Xd,X):
    E = []
    print(len(X),X)
    print(len(Xd),Xd)
    for i in range(len(Xd)-1):
        E[i] = X[i] - Xd[i]
        prop(E,0.5)
        jacob(X)
        mvt_Robot(X,Xd)
        robot.forwardKinematics(Xd)


#Passage de X dans un proportionnel
def prop(X,p):
	for i in X:
		i = i*p

#Passade de X dans une jacobienne
def jacob(X):
    j = pin.jacobianSubtreeCenterOfMass(model,data,2)
    for i in range(3) :
        if(j[i][0] != 0.):
            X[i] *= j[i]

def mvt_Robot(X,Xd):
    Xd = Xd + X



#Main 
#currentDir = os.getcwd()
#os.chdir('../')
workingDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# urdf directory path
package_path = workingDir
urdf_path = package_path + '/robots/urdf/planar_2DOF_TCP.urdf'
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
ID_F = robot.model.getFrameId("tcp")
ID = robot.model.getJointId("tcp")-1
simulateurTraj(1000)