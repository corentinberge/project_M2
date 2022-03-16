from ctypes import sizeof
from pyexpat import model
from numpy import double, linalg, math, sign, sqrt
from numpy.core.fromnumeric import shape
from numpy.lib.nanfunctions import _nanmedian_small
#from Robot.Yaskawa.Code.src.identification_YASKAWA.Trajectoire_yaskawa_v2 import Q_total
from pinocchio.visualize import GepettoVisualizer
from pinocchio.robot_wrapper import RobotWrapper
import matplotlib.pyplot as plt
import scipy.linalg as sp
import pinocchio as pin
import numpy as np
import os
from typing import Optional
from typing import Optional
import qpsolvers
from time import sleep

pre_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
package_path = pre_path + '/Modeles'
urdf_path = package_path + '/motoman_hc10_support/urdf/hc10_FGV.urdf'


robot = RobotWrapper()
robot.initFromURDF(urdf_path, package_path, verbose=True)
robot.initViewer(loadModel=True)
robot.display(robot.q0)

NJOINT = robot.model.njoints  # number of links
data = robot.data
model = robot.model



def Generate_posture_static():
    
    # Q_total=[[],[],[],[],[],[]]
    Q_total=[]
    Q_total=np.array(Q_total)
    posture1=np.array([[0],[0],[0],[0],[0],[0]])
    Q_total=posture1
    #print("shape of posture 1",np.array(posture1).shape)

    #print("shape of Q",Q_total.shape)
    Q_total=np.concatenate([Q_total,posture1], axis=1)

    posture2=np.array([[0],[math.pi/2],[-math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture2], axis=1)
    
    posture3=np.array([[0],[0],[-math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture3], axis=1)

    posture4=np.array([[0],[0],[math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture4], axis=1)

    posture5=np.array([[0],[0],[-math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture5], axis=1)
    
    posture6=np.array([[0],[0],[math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture6], axis=1)
    
    posture7=np.array([[0],[0],[math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture7], axis=1)

    posture8=np.array([[0],[0],[-math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture8], axis=1)

    
    posture9=np.array([[0],[-math.pi/2],[math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture9], axis=1)


    posture10=np.array([[0],[math.pi/4],[-math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture10], axis=1)

    
    posture11=np.array([[0],[math.pi/4],[math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture11], axis=1)

    
    posture12=np.array([[0],[math.pi/4],[-math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture12], axis=1)

    
    posture13=np.array([[0],[math.pi/4],[math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture13], axis=1)


    posture14=np.array([[0],[math.pi/4],[math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture14], axis=1)

    posture15=np.array([[0],[math.pi/4],[-math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture15], axis=1)

    posture16=np.array([[0],[math.pi/4],[math.pi],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture16], axis=1)

    posture17=np.array([[0],[-math.pi/4],[-math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture17], axis=1)

    posture18=np.array([[0],[-math.pi/4],[math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture18], axis=1)

    posture19=np.array([[0],[-math.pi/4],[-math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture19], axis=1)
    
    posture20=np.array([[0],[-math.pi/4],[math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture20], axis=1)
    
    posture21=np.array([[0],[-math.pi/4],[math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture21], axis=1)

    posture22=np.array([[0],[-math.pi/4],[-math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture22], axis=1)

    posture23=np.array([[0],[-math.pi/4],[math.pi],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture23], axis=1)

    posture24=np.array([[0],[-math.pi/3],[math.pi],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture24],axis=1)
    
    posture25=np.array([[0],[-math.pi/3],[-math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture25],axis=1)
    
    posture26=np.array([[0],[-math.pi/3],[math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture26],axis=1)
    
    posture27=np.array([[0],[-math.pi/3],[-math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture27],axis=1)
    
    posture28=np.array([[0],[-math.pi/3],[math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture28],axis=1)
    
    posture29=np.array([[0],[-math.pi/3],[math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture29],axis=1)
    
    posture30=np.array([[0],[-math.pi/3],[-math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture30],axis=1)
    
    posture31=np.array([[0],[-math.pi/3],[math.pi],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture31],axis=1)
    
    posture32=np.array([[0],[-math.pi/6],[math.pi],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture32],axis=1)
    
    posture33=np.array([[0],[-math.pi/6],[-math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture33],axis=1)
    
    posture34=np.array([[0],[-math.pi/6],[math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture34],axis=1)
    
    posture35=np.array([[0],[-math.pi/6],[-math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture35],axis=1)
    
    posture36=np.array([[0],[-math.pi/6],[math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture36],axis=1)
    
    posture37=np.array([[0],[-math.pi/6],[math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture37],axis=1)
    
    posture38=np.array([[0],[-math.pi/6],[-math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture38],axis=1)
    
    posture39=np.array([[0],[-math.pi/6],[math.pi],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture39],axis=1)
    
    posture40=np.array([[0],[math.pi/3],[math.pi],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture40],axis=1)
    
    posture41=np.array([[0],[math.pi/3],[-math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture41],axis=1)
    
    posture42=np.array([[0],[math.pi/3],[math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture42],axis=1)
    
    posture43=np.array([[0],[math.pi/3],[-math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture43],axis=1)
    
    posture44=np.array([[0],[math.pi/3],[math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture44],axis=1)
    
    posture45=np.array([[0],[math.pi/3],[math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture45],axis=1)
    
    posture46=np.array([[0],[math.pi/3],[-math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture46],axis=1)
    
    posture47=np.array([[0],[math.pi/3],[math.pi],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture47],axis=1)
    
    posture48=np.array([[0],[math.pi/6],[math.pi],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture48],axis=1)
    
    posture49=np.array([[0],[math.pi/6],[-math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture49],axis=1)
    
    posture50=np.array([[0],[math.pi/6],[math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture50],axis=1)
    
    posture51=np.array([[0],[math.pi/6],[-math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture51],axis=1)
    
    posture52=np.array([[0],[math.pi/6],[math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture52],axis=1)
    
    posture53=np.array([[0],[math.pi/6],[math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture53],axis=1)
    
    posture54=np.array([[0],[math.pi/6],[-math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture54],axis=1)
    
    posture55=np.array([[0],[math.pi/6],[math.pi],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture55],axis=1)

    posture56=np.array([[0],[math.pi/6],[math.pi],[math.pi/2],[3],[0]])
    Q_total=np.concatenate([Q_total,posture56],axis=1)

    posture57=np.array([[0],[math.pi/6],[math.pi],[math.pi/4],[3],[0]])
    Q_total=np.concatenate([Q_total,posture57],axis=1)


    return Q_total

def Generate_inertial_parameter():
    names = []
    for i in range(1, NJOINT):
        names += ['m'+str(i), 'mx'+str(i), 'my'+str(i), 'mz'+str(i)]
    print(names)
    phi = []
    for i in range(1, NJOINT):
        phi.extend(model.inertias[i].toDynamicParameters())

    #print('shape of phi:\t', np.array(phi).shape)   
    return names,phi

def Generate_Regression_vector(Q):
    nbSamples = Q.shape[1]
    #print(nbSamples)
    W_reg=[]
    dq_pin = np.zeros((6, 1))
    ddq_pin = np.zeros((6, 1))

    for i in range(nbSamples):
        W_reg.extend(pin.computeJointTorqueRegressor(model, data, Q[:, i], dq_pin[:, 0], ddq_pin[:, 0]))
    
    W_reg=np.array(W_reg)
    print('shape of Wreg',np.array(W_reg).shape)
    return W_reg

def Redimention_Regression_vector(W_reg_Pin):
    newR = np.delete(W_reg_Pin, np.s_[54:60], axis = 1)
    newR = np.delete(newR, np.s_[44:50], axis = 1)
    newR = np.delete(newR, np.s_[34:40], axis = 1)
    newR = np.delete(newR, np.s_[24:30], axis = 1)
    newR = np.delete(newR, np.s_[14:20], axis = 1)
    newR = np.delete(newR, np.s_[4:10], axis = 1)
    return newR

def Generate_text(neW_reg):
    f = open( 'New_regresseur.txt','w')
    for i in range(348):
        for j in range(24):
            text = str(neW_reg[i,j])
            f.writelines(text)
            f.write(', ')
        f.write('\n')
    f.close()

if __name__=="__main__":
    Q_pos=[]
    Q_pos=Generate_posture_static()
    #print('shape of Q',np.array(Q).shape)
    for i in range(Q_pos[0].size):
        robot.display(Q_pos[:,i])
        sleep(0.1)
    print('shape of Q',np.array(Q_pos).shape)
# ========== Step 2 - generate inertial parameters for all links (excepted the base link)
    names,phi = Generate_inertial_parameter()

# ========== Step 3- Create IDM with pinocchio (regression matrix)
    W_reg_Pin = Generate_Regression_vector(Q_pos)
    #print('shape of Q',np.array(Q).shape)
# ========== Step 4- Redim regression vector (no dq,ddq)   
    neW_reg = Redimention_Regression_vector(W_reg_Pin)
    print('shape of neW_reg',np.array(neW_reg).shape)
    print(neW_reg)
    Generate_text(neW_reg)
    
