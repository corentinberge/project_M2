import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np 
import math
from time import sleep



def load_model():
    #change access path ! 
    path = '/home/jo/'
    urdf = '/home/jo/robots/planar_2DOF/urdf/planar_2DOF.urdf'
    robot = RobotWrapper()
    robot.initFromURDF(urdf,path,verbose=True)
    robot.initViewer(loadModel=True)
    robot.display(robot.q0)
    return robot


robot = load_model()
NQ = robot.nq #number of joint angle 
NV = robot.nv#number of joint vélocity 
NJOINT = robot.model.njoints
gv = robot.viewer.gui


q = np.zeros(NQ) # initial config of the robot
robot.display(q)

for i in range(1000):
    tmp = math.sin(i*0.01)
    for j in range(NJOINT-1):
        q[j] = tmp
    robot.display(q)
    sleep(0.01)


print("press enter to continue")
input()
gv.deleteNode('world', True)  # name, all=True
