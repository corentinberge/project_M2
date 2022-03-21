from ctypes import sizeof
from operator import index
from pyexpat import model
from random import random
from termios import TCSAFLUSH
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
import random

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
NQ = robot.nq     

index_vector_to_delete=[]
indexQR=0

def Generate_posture_static():
    
    # Q_total=[[],[],[],[],[],[]]
    Q_total=[]
    Q_total=np.array(Q_total)
    posture1=np.array([[0],[0],[0],[0],[0],[0]])
    Q_total=posture1
    #print("shape of posture 1",np.array(posture1).shape)

    posture3=np.array([[0],[0],[-math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture3], axis=1)

    posture100=np.array([[-0.959931],[-0.313159],[1.69297],[0.05],[-1.98968],[0.959931]])
    Q_total=np.concatenate([Q_total,posture100], axis=1)

    posture101=np.array([[math.pi],[0],[math.pi],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture101], axis=1)

    posture102=np.array([[2.356125],[0],[math.pi],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture102], axis=1)

    posture54=np.array([[0],[math.pi/6],[-math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture54],axis=1)

    posture4=np.array([[0],[0],[math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture4], axis=1)

    posture5=np.array([[0],[0],[-math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture5], axis=1)

    posture103=np.array([[2.356125],[-0.25],[math.pi],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture103], axis=1)

    
    posture12=np.array([[0],[math.pi/4],[-math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture12], axis=1)

    posture6=np.array([[0],[0],[math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture6], axis=1)
    
    posture104=np.array([[2.356125],[0.6],[math.pi],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture104], axis=1)

    posture7=np.array([[0],[0],[math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture7], axis=1)

    posture8=np.array([[0],[0],[-math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture8], axis=1)

    posture105=np.array([[math.pi],[0],[4.71225],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture105], axis=1)

    posture9=np.array([[0],[-math.pi/2],[math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture9], axis=1)

    posture10=np.array([[0],[math.pi/4],[-math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture10], axis=1)
   
    posture106=np.array([[math.pi],[0],[1.57075],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture106], axis=1)

    posture11=np.array([[0],[math.pi/4],[math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture11], axis=1)

    posture13=np.array([[0],[math.pi/4],[math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture13], axis=1)

    posture14=np.array([[0],[math.pi/4],[math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture14], axis=1)

    posture107=np.array([[math.pi],[0],[1.9],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture107], axis=1)

    posture15=np.array([[0],[math.pi/4],[-math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture15], axis=1)

    posture16=np.array([[0],[math.pi/4],[math.pi],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture16], axis=1)

    posture108=np.array([[math.pi],[0],[math.pi],[math.pi],[0],[0]])
    Q_total=np.concatenate([Q_total,posture108], axis=1)

    posture17=np.array([[0],[-math.pi/4],[-math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture17], axis=1)

    posture18=np.array([[0],[-math.pi/4],[math.pi/2],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture18], axis=1)

    posture109=np.array([[math.pi],[0],[math.pi],[-math.pi],[0],[0]])
    Q_total=np.concatenate([Q_total,posture109], axis=1)

    posture19=np.array([[0],[-math.pi/4],[-math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture19], axis=1)
    
    posture20=np.array([[0],[-math.pi/4],[math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture20], axis=1)
    
    posture21=np.array([[0],[-math.pi/4],[math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture21], axis=1)

    posture110=np.array([[math.pi],[0],[math.pi],[0],[-math.pi],[0]])
    Q_total=np.concatenate([Q_total,posture110], axis=1)

    posture22=np.array([[0],[-math.pi/4],[-math.pi/(1.5)],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture22], axis=1)

    posture23=np.array([[0],[-math.pi/4],[math.pi],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture23], axis=1)

    posture111=np.array([[math.pi],[0],[math.pi],[0],[math.pi],[0]])
    Q_total=np.concatenate([Q_total,posture111], axis=1)

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
    
    posture5=np.array([[0],[0],[-math.pi/4],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture5], axis=1)
    
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
        names += ['m'+str(i), 'mx'+str(i), 'my'+str(i), 'mz'+str(i), 'Ixx'+str(i),
              'Ixy'+str(i), 'Iyy'+str(i), 'Izx'+str(i), 'Izy'+str(i), 'Izz'+str(i)] 
    phi = []
    for i in range(1, NJOINT):
        phi.extend(model.inertias[i].toDynamicParameters())

    #print('shape of phi:\t', np.array(phi).shape)   
    return names,phi

def Generate_Regression_vector(Q,nbSamples):
    W_reg=[]
    dq_pin = np.zeros((NQ, nbSamples))# matrice des zero 6 lignes nbr de posture en colones 
    ddq_pin = np.zeros((NQ, nbSamples)) # matrice des zero 6 lignes nbr de posture en colones
    for i in range(nbSamples):
        W_reg.extend(pin.computeJointTorqueRegressor(model, data, Q[:, i], dq_pin[:, i], ddq_pin[:, i]))
    
    W_reg=np.array(W_reg)

    tau= []
    for i in range(nbSamples):
        tau.extend(pin.rnea(model, data, Q[:, i], dq_pin[:, i], ddq_pin[:, i]))
    
    tau=np.array(tau)
    tau=np.double(tau)

    print('Shape of Wreg',np.array(W_reg).shape)
    print('Shape of tau:\t', np.array(tau).shape)
    return W_reg,tau

def Redimention_Regression_vector(W_reg_Pin,phi,names):
    threshold = 0.000001
## we have just to change the w_modified so we can work with the different input/output
#  w is calculated with the input/output from than's data file
#  W_pin is calculated with the input/output generated by pinocchio


    W_modified = np.array(W_reg_Pin[:])
    #index_vector_to_delete=[]
    
    tmp = []
    for i in range(len(phi)):
        if (np.dot([W_modified[:, i]], np.transpose([W_modified[:, i]]))[0][0] <= threshold):
            tmp.append(i)
            index_vector_to_delete.append(i)
    tmp.sort(reverse=True)
    
    # index_vector_to_delete.sort(reverse=True)

    phi_modified = phi[:]
    names_modified = names[:]
    #for i in vector to delete:
        # W_modified = np.delete(W_modified, i, 1)
        # phi_modified = np.delete(phi_modified, i, 0)
        # names_modified = np.delete(names_modified, i, 0)




    for i in tmp:
        W_modified = np.delete(W_modified, i, 1)
        phi_modified = np.delete(phi_modified, i, 0)
        names_modified = np.delete(names_modified, i, 0)

    print('shape of W_m:\t', W_modified.shape)
    print('shape of phi_m:\t', np.array(phi_modified).shape)

    print('the index vector to delete',index_vector_to_delete)
    return W_modified,phi_modified,names_modified

def Calculate_base_param(Q, R, P,Tau):
    tmp = 0
    #indexQR=0
    for i in range(np.diag(R).shape[0]):
            if abs(np.diag(R)[i]) < 0.000001:
                tmp = i
                #indexQR=i

    print('index QR <3<3<3<3<3<3<3<3<3<3<3<3<3<3',indexQR)
    R1 = R[:tmp, :tmp]
    R2 = R[:tmp, tmp:]

    Q1 = Q[:, :tmp]
    
    for i in (tmp, len(P)-1):
        names.pop(P[i])
    
    print('Shape of R1:\t', np.array(R1).shape)
    print('Shape of R2:\t', np.array(R2).shape)
    print('Shape of Q1:\t', np.array(Q1).shape)

    beta = np.dot(sp.pinv(R1), R2)
    print('Shape of beta:\t', np.array(beta).shape)

    phi_base = np.dot(np.linalg.inv(R1), np.dot(Q1.T,Tau))  # Base parameters
    W_base = np.dot(Q1, R1)                             # Base regressor
    print('Shape of W_base:\t', np.array(W_base).shape)
    print('Shape of Phi_base:\t', np.array(phi_base).shape)
    
    return R1,R2,Q1,W_base,phi_base

def Redimention_x2(Q_posture):
    Exp_Regrs,T = Generate_Regression_vector(Q_posture,Q_posture.shape[1])
    Regresseur =  np.array(Exp_Regrs)
    Regresseur_Redim = np.delete(Regresseur,index_vector_to_delete, axis=1)
    (Q_1, R_1, P_1) = sp.qr(Regresseur_Redim, pivoting=True)
    R1_1,R1_1,Q1_1,W_base1,phi_base1 = Calculate_base_param(Q_1, R_1, P_1,T)
    #Regresseur = np.delete(Regresseur,indexQR, axis=1)
    print ('Shape of Regresseur',np.array(Regresseur).shape)
    return W_base1

def Conditionnement(M_for_Cond,Q_posture):

    M_for_Cond = np.array(M_for_Cond[0:18])
    print ('1111shape of',np.array(M_for_Cond).shape)
    Q_posture = np.array(Q_posture)
    #Genrate regressor matric form experiences
    Redim_Exp_Regrs = Redimention_x2(Q_posture)
    Redim_Exp_Regrs = np.array(Redim_Exp_Regrs)
    #======================================
    print ('shape of Redim_Exp_Regrs',np.array(Redim_Exp_Regrs).shape)
    print("======2======")
    Cond = []
    Exps = []
    for i in range(Q_pos.shape[1]):  
        AddExp = np.array(Redim_Exp_Regrs[i*6:i*6+6])
        #print ('shape of AddExp',i," :",np.array(AddExp).shape)
        M_for_Cond = np.concatenate((M_for_Cond,AddExp))# je suis pas dacc
        #print ('shape of M_for_Cond',i," :",np.array(M_for_Cond).shape)
        Exps.append(i)
        Cond.append(linalg.cond(M_for_Cond),np.inf)

    #print ('shape of Exps',np.array(Exps).shape)
    #print ('shape of Cond',np.array(Cond).shape)
    return Cond,Exps 

def Generate_text(neW_reg):
    f = open('New_regresseur.txt','w')
    for i in range(neW_reg.shape[1]):
        for j in range(20):
            text = str(neW_reg[i,j])
            f.writelines(text)
            f.write(', ')
        f.write('\n')
    f.close()

#=============================================================================
#=============================================================================
if __name__=="__main__": 
    Q_pos=[]
    Q_pos = Generate_posture_static()
    random.shuffle(Q_pos)

    #print('shape of Q',np.array(Q).shape)
    for i in range(Q_pos[0].size):
        robot.display(Q_pos[:,i])
        sleep(0.1)
    print('shape of Q',np.array(Q_pos).shape)

# ========== Step 2 - generate inertial parameters for all links (excepted the base link)
    names,phi = Generate_inertial_parameter()

# ========== Step 3- Create IDM with pinocchio (regression matrix)
    # Generate a regressor matrix with random value
    nbSample =500
    q_random= np.random.rand(NQ, nbSample) * np.pi - np.pi/2
    W_reg_Pin,tau = Generate_Regression_vector(q_random,nbSample)
    #print('shape of Q',np.array(Q).shape)

# ========== Step 4- Redim regression vector (no dq,ddq)   
    W_modified,phi_modified,names_modified = Redimention_Regression_vector(W_reg_Pin,phi,names)
    print('shape of W_modified',np.array(W_modified).shape)
    print(names_modified)

# ========== Step 5 - QR decomposition + pivoting
    (Q, R, P) = sp.qr(W_modified, pivoting=True)
    print('shape of Q:\t', np.array(Q).shape)
    print('shape of R:\t', np.array(R).shape)
    print('shape of P:\t', np.array(P).shape)

# ========== Step 6 - Calculate base parameters /Calculate the Phi modified
    R1,R2,Q1,W_base,phi_base = Calculate_base_param(Q, R, P,tau)

# ========== Step 7 - Conditionnement with exp add.
    #Cond, Nbr_exp = Conditionnement(Q1,Q_pos)
    Condi = linalg.cond(W_base)
    print('Voila',Condi)

# ========== Step 8 - Affichage with exp add.    
    
    Nbr_exp = np.delete(Nbr_exp,[0])
    plt.figure('Conditionnement en fonction du nombre dexps')
    plt.plot(Nbr_exp, Cond, 'g', linewidth=2, label='tau')
    plt.title('Conditionnement en fonction du nombre dexps')
    plt.xlabel('NbrsExps')
    plt.ylabel('Cond')
    plt.legend()
    plt.show()
#   print(names)
