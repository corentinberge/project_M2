from cmath import tau
from contextlib import suppress
import math
from sqlite3 import Time
from tkinter import W
from numpy import double, linalg, sign, size, sqrt
from numpy.core.fromnumeric import shape
from numpy.lib.nanfunctions import _nanmedian_small
from sklearn.metrics import precision_score
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
import scipy
from scipy import signal

np.set_printoptions(precision=150,suppress=True)

package_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + '/Modeles/'
urdf_path = package_path + 'motoman_hc10_support/urdf/hc10_FGV.urdf'

 # ========== Step 1 - load model, create robot model and create robot data

robot = RobotWrapper()
robot.initFromURDF(urdf_path, package_path, verbose=True)
robot.initViewer(loadModel=True)
robot.display(robot.q0)

data = robot.data
model = robot.model
NQ = robot.nq                 # joints angle
NV = robot.nv                 # joints velocity
NJOINT = robot.model.njoints  # number of links
gv = robot.viewer.gui

#sampling time 
Tech=(1/10)
# INITIALISATION 

# deg5 est une marge pour eviter de depasser les butees min max des articulations
deg_5=-0.08726646259971647
deg_5=deg_5+0.25*deg_5

# position initial des articulations butees min
q_start=[-3.141592653589793-1*deg_5,-3.141592653589793/4-1*deg_5, -0.08726646259971647-1*deg_5,-3.141592653589793-1*deg_5, -3.141592653589793-1*deg_5, -3.141592653589793-1*deg_5]

# position final des articulations butees max
q_end=[3.141592653589793, 3.141592653589793/4, 6.19591884457987, 3.141592653589793, 3.141592653589793, 3.141592653589793]

# vitesse et acceleration max des articulations
Vmax=[2.2689280275926285, 2.2689280275926285, 3.141592653589793, 3.141592653589793, 4.36, 4.36]
acc_max=[4,4,4,4,4,4]

# dictionnaire des parametres
param={
        'nb_iter_OEM':2, # number of OEM to be optimized
        'tf':1,# duration of one OEM
        'freq': 1, # frequency of the fourier serie coefficient
        'nb_repet_trap': 2, # number of repetition of the trapezoidal motions
        'q_safety': 0.08, # value in radian (=5deg) to remove from the actual joint limits
        'q_lim_def': 1.57,# default value for joint limits in cas the URDF does not have the info
        'dq_lim_def':4, # in rad.s-1
        'ddq_lim_def':20, # in rad.s-2
        'tau_lim_def':3, # in N.m
        'trapez_vel_steps':20,# Velocity step in % of the max velocity
        'ts_OEM':1/100,# Sampling frequency of the optimisation process
        'ts':1/1000,# Sampling frequency of the trajectory to be recorded
        'Friction':True,
        'fv':0.01,# default values of the joint viscous friction in case they are used
        'fc':0.1,# default value of the joint static friction in case they are used
        'K_1':1, # default values of drive gains in case they are used
        'K_2':2, # default values of drive gains in case they are used
        'nb_harmonics': 2,# number of harmonics of the fourier serie
        'mass_load':3.0,
        'idx_base_param':(1, 3, 6, 11, 13, 16),# retrieved from previous analysis
        'sync_joint_motion':0, # move all joint simultaneously when using quintic polynomial interpolation
        'eps_gradient':1e-6,# numerical gradient step
        'ANIMATE':0,# plot flag for gepetto-viewer
        'SAVE_FILE':0
    }
param['NbSample']=int (param['tf']/param['ts'])

# Joint configuration pour l'interpolation polinomiale

Jcf_Home=np.array([
                [3.141592653589793/4,0,0],
                [3.141592653589793/4,0,0],
                [1.69297,0,0],
                [3.141592653589793/4,0,0],
                [-1.98968,0,0],
                [0.959931 ,0,0],
                ])

Jci_avBute=np.array([
                [0 ,0,0],
                [0 ,0,0],
                [3.141592653589793,0,0],
                [0 ,0,0],
                [0 ,0,0],
                [0 ,0,0],
                ])

Jcf_avBute=np.array([
                [-3.141592653589793-1*deg_5 ,0,acc_max[0]],
                [-3.141592653589793/4-1*deg_5,0,acc_max[1]],
                [ -0.08726646259971647-1*deg_5,0,acc_max[2]],
                [-3.141592653589793-1*deg_5 ,0,acc_max[3]],
                [-3.141592653589793-1*deg_5 ,0,acc_max[4]],
                [-3.141592653589793-1*deg_5 ,0,acc_max[5]],
                ])

Jci_aprBute=np.array([
                [-3.141592653589793-1*deg_5,0,-acc_max[0]],
                [-3.141592653589793/4-1*deg_5,0,-acc_max[1]],
                [ -0.08726646259971647-1*deg_5,0,acc_max[2]],
                [-3.141592653589793-1*deg_5,0,-acc_max[3]],
                [-3.141592653589793-1*deg_5,0,-acc_max[4]],
                [-3.141592653589793-1*deg_5,0,-acc_max[5]],
                ])

Jcf_aprBute=np.array([
                [0 ,0,0],
                [0 ,0,0],
                [math.pi ,0,0],
                [0 ,0,0],
                [0 ,0,0],
                [0 ,0,0],
                ])

def generation_palier_vitesse_calcul_oneJoint(nbr_rep,prct,q_start,q_end,Vmax,acc_max,Tech):
    #this function take in input :1- the number of repetition of a joint
    #                             2- the data of the chosen joint motion(q_start q_end V acc)
                                # this function provide the option to chose a specific percentage of the max velocity
    
    # and return the data of the chosen joint in a matrix that combine:
    #                                                       1- position vector after  repetion 
    #                                                       2- velosity vector after  repetion 
    #                                                       3- acc vector after  repetion 
    #                                                       4- time vector with after repetion     
   
    vitesse=prct*Vmax
    prct_var=prct
    i=0
    loop=0 # pour un test dans ma tete
    Q_all=[]
    Q=[]
    V=[]
    T=[]
    tf=0
    A=[]
    while(vitesse<=Vmax):
        Q_palier=calcul_Q_all_variable_a2a(nbr_rep,q_start,q_end,vitesse,acc_max,Tech)
        #Q_all.append(Q)
        prct_var+=prct
        vitesse=prct_var*Vmax
        loop+=1
        # print('avant trajectory')      
        q=Q_palier[0]
        t=Q_palier[1]
        v=Q_palier[2]
        a=Q_palier[3]
        # print('apres trajectory')  
        Q.extend(q)
        V.extend(v)
        A.extend(a)
        for i in range(np.array(t).size):
            t[i]+=tf

        T.extend(t)       
        tf=T[np.array(T).size-1]

    Q_all.append(Q)
    Q_all.append(T)
    Q_all.append(V)
    Q_all.append(A)

    return Q_all

def generation_palier_vitesse_calcul_allJoint(nbr_rep,prct,nbr_joint,q_start,q_end,Vmax,acc_max,Tech,padind_vect_chandell):
    # this function return the position velocity acc for all joint one by one 
    # the pading vector is to put the joint that dont move in a specific position

    tf=0
    T=[]
    Q_palier_V_Joint=[]
    Q_total_one_joint=[]
    V_total_one_joint=[]
    A_total_one_joint=[]
    Q_total_All_Joint=[[],[],[],[],[],[]]
    V_total_All_Joint=[[],[],[],[],[],[]]
    A_total_All_Joint=[[],[],[],[],[],[]]

    # print('Vmax[0]',Vmax[0])
    
    NbSample_interpolate=100
    # j=1
    for i in range(nbr_joint):

        Q_palier_V_Joint=generation_palier_vitesse_calcul_oneJoint(nbr_rep,prct,q_start[i],q_end[i],Vmax[i],acc_max[i],Tech)
        Q_total_one_joint,V_total_one_joint,A_total_one_joint=calcul_QVA_joints_total(nbr_joint,i,Q_palier_V_Joint,padind_vect_chandell)
        Q_total_All_Joint=np.concatenate([Q_total_All_Joint,Q_total_one_joint], axis=1)
        V_total_All_Joint=np.concatenate([V_total_All_Joint,V_total_one_joint], axis=1)
        A_total_All_Joint=np.concatenate([A_total_All_Joint,A_total_one_joint], axis=1)


        t=Q_palier_V_Joint[1]
        times=t
        for j in range(np.array(t).size):
            t[j]+=tf
        T.extend(t)       
        tf=T[np.array(T).size-1]

        
        q1,dq1,ddq1,times=generateQuinticPolyTraj_version_GF(Jci_aprBute[i],Jcf_aprBute[i],Vmax[i]-0.2,Tech)
        # q1,dq1,ddq1=Bang_Bang_acceleration_profile(Jci2[i][0],Jcf2[i][0],Jci2[i][1],Jci2[i][2],Tech)
        Q_inter1_one_joint,V_inter1_one_joint,A_inter1_one_joint=calcul_QVA_joints_total(nbr_joint,i,[q1,times,dq1,ddq1],padind_vect_chandell)
        Q_total_All_Joint=np.concatenate([Q_total_All_Joint,Q_inter1_one_joint], axis=1)
        V_total_All_Joint=np.concatenate([V_total_All_Joint,V_inter1_one_joint], axis=1)
        A_total_All_Joint=np.concatenate([A_total_All_Joint,A_inter1_one_joint], axis=1)
        t=times

        for k in range(np.array(t).size):
            t[k]+=tf
        T.extend(t)       
        tf=T[np.array(T).size-1]

        if(i<nbr_joint-1):
            q,dq,ddq,times=generateQuinticPolyTraj_version_GF(Jci_avBute[i+1],Jcf_avBute[i+1],Vmax[i+1]-0.2,Tech)
            # q,dq,ddq=Bang_Bang_acceleration_profile(Jci1[i+1][0],Jcf1[i+1][0],Jci1[i+1][1],Jci1[i+1][2],Tech)
            Q_inter2_one_joint,V_inter2_one_joint,A_inter2_one_joint=calcul_QVA_joints_total(nbr_joint,i+1,[q,times,dq,ddq],padind_vect_chandell)
            Q_total_All_Joint=np.concatenate([Q_total_All_Joint,Q_inter2_one_joint], axis=1)
            V_total_All_Joint=np.concatenate([V_total_All_Joint,V_inter2_one_joint], axis=1)
            A_total_All_Joint=np.concatenate([A_total_All_Joint,A_inter2_one_joint], axis=1)
            # j=j+1
        t=times

        for l in range(np.array(t).size):
            t[l]+=tf
        T.extend(t)       
        tf=T[np.array(T).size-1]
        # plot_QVA_total(times,nbr_joint,Q_inter_one_joint,V_inter_one_joint,A_inter_one_joint,'max_')
    
            
    print('shape of Time vect T',np.array(T).shape)
    print('Q_total_All_Joint',np.array(Q_total_All_Joint).shape)
    return T,Q_total_All_Joint,V_total_All_Joint,A_total_All_Joint
    
def trajectory_axe2axe_palier_de_vitesse_one_joint():
# this function can operate in two mode it can generate increasing velocity for each joint upon request
# and it can generate increasing velosity for all joint(one by one =>joint after joint)

    # data from user
    print('enter the number of Yaskawa joint you want to move (counting start from 0) ')
    joint_i=int(input())
    print('entre the percentage of increasing velosity')
    prct=float(input())
    print('enter the number of repetition of the Yaskawa joints')
    nbr_rep=int(input())
    nbr_joint=6 #yaskawa case
    
    Q_total_All_Joint=[[],[],[],[],[],[]]
    V_total_All_Joint=[[],[],[],[],[],[]]
    A_total_All_Joint=[[],[],[],[],[],[]]

    #mode1: generating trajectorys with increasing velocity for each joint upon request
    Jcf_Home0=Jcf_Home[0][0]
    Jcf_Home1=Jcf_Home[1][0]
    Jcf_Home2=Jcf_Home[2][0]
    Jcf_Home3=Jcf_Home[3][0]
    Jcf_Home4=Jcf_Home[4][0]
    Jcf_Home5=Jcf_Home[5][0]

    pading_vect=[
                [Jcf_Home0,Jcf_Home1,Jcf_Home2,Jcf_Home3,Jcf_Home4,Jcf_Home5],
                [0,Jcf_Home1,Jcf_Home2,Jcf_Home3,Jcf_Home4,Jcf_Home5],
                [0,0,Jcf_Home2,Jcf_Home3,Jcf_Home4,Jcf_Home5],
                [0,0,math.pi,Jcf_Home3,Jcf_Home4,Jcf_Home5],
                [0,0,math.pi,0,Jcf_Home4,Jcf_Home5],
                [0,0,math.pi,0,0,Jcf_Home5],
                ]
    
    padind_vect_chandell=[0,0,math.pi,0,0,0]

    # joint3=2
    # q,dq,ddq,times=generateQuinticPolyTraj_version_GF(Jcf_Homme[joint3],Jci_avBute[joint3],Vmax[joint3],Tech)
    # Q_inter_Home_joint,V_inter_Home_joint,A_inter_Home_joint=calcul_QVA_joints_total(nbr_joint,joint3,[q,times,dq,ddq])
    # Q_total_All_Joint=np.concatenate([Q_total_All_Joint,Q_inter_Home_joint], axis=1)
    # V_total_All_Joint=np.concatenate([V_total_All_Joint,V_inter_Home_joint], axis=1)
    # A_total_All_Joint=np.concatenate([A_total_All_Joint,A_inter_Home_joint], axis=1)

    for i in range(nbr_joint):
        q,dq,ddq,times=generateQuinticPolyTraj_version_GF(Jcf_Home[i],Jci_avBute[i],Vmax[i],Tech)
        Q_inter_Home_joint,V_inter_Home_joint,A_inter_Home_joint=calcul_QVA_joints_total(nbr_joint,i,[q,times,dq,ddq],pading_vect[i])
        Q_total_All_Joint=np.concatenate([Q_total_All_Joint,Q_inter_Home_joint], axis=1)
        V_total_All_Joint=np.concatenate([V_total_All_Joint,V_inter_Home_joint], axis=1)
        A_total_All_Joint=np.concatenate([A_total_All_Joint,A_inter_Home_joint], axis=1)
   
    
    q,dq,ddq,times=generateQuinticPolyTraj_version_GF(Jci_avBute[joint_i],Jcf_avBute[joint_i],Vmax[joint_i],Tech)
    Q_inter_Home_joint,V_inter_Home_joint,A_inter_Home_joint=calcul_QVA_joints_total(nbr_joint,joint_i,[q,times,dq,ddq],padind_vect_chandell)
    Q_total_All_Joint=np.concatenate([Q_total_All_Joint,Q_inter_Home_joint], axis=1)
    V_total_All_Joint=np.concatenate([V_total_All_Joint,V_inter_Home_joint], axis=1)
    A_total_All_Joint=np.concatenate([A_total_All_Joint,A_inter_Home_joint], axis=1)
             
    Q_pallier_vitesse=[]
    Q_pallier_vitesse = generation_palier_vitesse_calcul_oneJoint(nbr_rep,prct,q_start[joint_i],q_end[joint_i],Vmax[joint_i],acc_max[joint_i],Tech)
    Q_total_one_joint,V_total_one_joint,A_total_one_joint=calcul_QVA_joints_total(nbr_joint,joint_i,Q_pallier_vitesse,padind_vect_chandell)
    
    Q_total_All_Joint=np.concatenate([Q_total_All_Joint,Q_total_one_joint], axis=1)
    V_total_All_Joint=np.concatenate([V_total_All_Joint,V_total_one_joint], axis=1)
    A_total_All_Joint=np.concatenate([A_total_All_Joint,A_total_one_joint], axis=1)

    q,dq,ddq,times=generateQuinticPolyTraj_version_GF(Jci_aprBute[joint_i],Jcf_aprBute[joint_i],Vmax[joint_i],Tech)
    Q_inter_Home_joint,V_inter_Home_joint,A_inter_Home_joint=calcul_QVA_joints_total(nbr_joint,joint_i,[q,times,dq,ddq],padind_vect_chandell)
    Q_total_All_Joint=np.concatenate([Q_total_All_Joint,Q_inter_Home_joint], axis=1)
    V_total_All_Joint=np.concatenate([V_total_All_Joint,V_inter_Home_joint], axis=1)
    A_total_All_Joint=np.concatenate([A_total_All_Joint,A_inter_Home_joint], axis=1)
    T=[]

    # # plot_Trajectory(Q_pallier_vitesse)
    plot_QVA_total(T,nbr_joint,Q_total_All_Joint,V_total_All_Joint,A_total_All_Joint,'joint_')
    Generate_text_data_file_Q_txt(Q_total_All_Joint,V_total_All_Joint,A_total_All_Joint)
    # tau,w=Generate_Torque_Regression_matrix(nbr_joint,Q_total_All_Joint,V_total_All_Joint,A_total_All_Joint)
    # phi_etoile,tau_estime=estimation_with_qp_solver(w,tau)
    # print("shape of phi_etoile",phi_etoile.shape)

    return Q_total_All_Joint,V_total_All_Joint,A_total_All_Joint
    
def axe2axe_palier_de_vitesse_all_joint_one_by_one():
    #Mode 2: generating trajectory with increasing velosity for all joint(one by one =>joint after joint)
    
    # print('enter the number of Yaskawa joint you want to move (counting start from 0) ')
    # joint_i=int(input())
    print('entre the percentage of increasing velosity')
    prct=float(input())
    print('enter the number of repetition of the Yaskawa joints')
    nbr_rep=int(input())
    nbr_joint=6 #yaskawa case
    joint_i=0
    
    Q_total_All_Joint=[[],[],[],[],[],[]]
    V_total_All_Joint=[[],[],[],[],[],[]]
    A_total_All_Joint=[[],[],[],[],[],[]]

    #mode1: generating trajectorys with increasing velocity for each joint upon request
    Jcf_Home0=Jcf_Home[0][0]
    Jcf_Home1=Jcf_Home[1][0]
    Jcf_Home2=Jcf_Home[2][0]
    Jcf_Home3=Jcf_Home[3][0]
    Jcf_Home4=Jcf_Home[4][0]
    Jcf_Home5=Jcf_Home[5][0]

    pading_vect=[
                [Jcf_Home0,Jcf_Home1,Jcf_Home2,Jcf_Home3,Jcf_Home4,Jcf_Home5],
                [0,Jcf_Home1,Jcf_Home2,Jcf_Home3,Jcf_Home4,Jcf_Home5],
                [0,0,Jcf_Home2,Jcf_Home3,Jcf_Home4,Jcf_Home5],
                [0,0,math.pi,Jcf_Home3,Jcf_Home4,Jcf_Home5],
                [0,0,math.pi,0,Jcf_Home4,Jcf_Home5],
                [0,0,math.pi,0,0,Jcf_Home5],
                ]
    
    padind_vect_chandell=[0,0,math.pi,0,0,0]

    # joint3=2
    # q,dq,ddq,times=generateQuinticPolyTraj_version_GF(Jcf_Homme[joint3],Jci_avBute[joint3],Vmax[joint3],Tech)
    # Q_inter_Home_joint,V_inter_Home_joint,A_inter_Home_joint=calcul_QVA_joints_total(nbr_joint,joint3,[q,times,dq,ddq])
    # Q_total_All_Joint=np.concatenate([Q_total_All_Joint,Q_inter_Home_joint], axis=1)
    # V_total_All_Joint=np.concatenate([V_total_All_Joint,V_inter_Home_joint], axis=1)
    # A_total_All_Joint=np.concatenate([A_total_All_Joint,A_inter_Home_joint], axis=1)

    for i in range(nbr_joint):
        q,dq,ddq,times=generateQuinticPolyTraj_version_GF(Jcf_Home[i],Jci_avBute[i],Vmax[i],Tech)
        Q_inter_Home_joint,V_inter_Home_joint,A_inter_Home_joint=calcul_QVA_joints_total(nbr_joint,i,[q,times,dq,ddq],pading_vect[i])
        Q_total_All_Joint=np.concatenate([Q_total_All_Joint,Q_inter_Home_joint], axis=1)
        V_total_All_Joint=np.concatenate([V_total_All_Joint,V_inter_Home_joint], axis=1)
        A_total_All_Joint=np.concatenate([A_total_All_Joint,A_inter_Home_joint], axis=1)

    q,dq,ddq,times=generateQuinticPolyTraj_version_GF(Jci_avBute[0],Jcf_avBute[joint_i],Vmax[joint_i],Tech)
    Q_inter_Home_joint,V_inter_Home_joint,A_inter_Home_joint=calcul_QVA_joints_total(nbr_joint,joint_i,[q,times,dq,ddq],padind_vect_chandell)
    Q_total_All_Joint=np.concatenate([Q_total_All_Joint,Q_inter_Home_joint], axis=1)
    V_total_All_Joint=np.concatenate([V_total_All_Joint,V_inter_Home_joint], axis=1)
    A_total_All_Joint=np.concatenate([A_total_All_Joint,A_inter_Home_joint], axis=1)

    T,Q_total_Joint,V_total_Joint,A_total_Joint=generation_palier_vitesse_calcul_allJoint(nbr_rep,prct,nbr_joint,q_start,q_end,Vmax,acc_max,Tech,padind_vect_chandell)
    Q_total_All_Joint=np.concatenate([Q_total_All_Joint,Q_total_Joint], axis=1)
    V_total_All_Joint=np.concatenate([V_total_All_Joint,V_total_Joint], axis=1)
    A_total_All_Joint=np.concatenate([A_total_All_Joint,A_total_Joint], axis=1)
    # Generate_text_data_file_Q_txt(Q_total_All_Joint)
    plot_QVA_total(T,nbr_joint,Q_total_All_Joint,V_total_All_Joint,A_total_All_Joint,'joint_')

def trajectory_mode_a2a_sync():
    #this function dont take an input data  but ask the user to enter his own data: 
    # it returns two mode of trajectorys:
    #   mode 1: axe to axe
    #   1-it return a specifique trajectory for a chosen joint
    #   and for other joints trajectory with zero value
    #   so other joints dont move
    #   2-it return the fricion force of coulomb
    
    # data.qlim did not work
    Q_total=[]
    V_total=[]
    A_total=[]
    Q_total_per_joint=[]
    mode=0
    while (mode!=1 and mode!=2):
        
        # entering data from user 

        print('enter 1 for the mode axe to axe, and 2 for the mode syncronized')
        mode=float(input())
        if(mode==1):
            print('enter the total number of joints')
            nbr_joint=int(input())

            print('enter the number of joint you want to move (counting start from 0) ')
            joint_i=float(input())

            print('enter lower bound position (q_min)')
            q_min=float(input())

            print('enter uper bound position (q_max)')
            q_max=float(input())
            
            print('enter the MAX velocity of joint')
            V_joint=float(input())

            print('enter the desire  prcentatge of the MAX velocity of joint')
            pourcentage=float(input())
            
            print('enter the acceleration of joint')
            acc_joint=float(input())

            print('enter number of repetition time of motion')
            nbr_rep=int(input())
            
            # calculs of position velosity acceleration for the chosen joint with variation 
            # of the Vmax 80% 60% 40% 20%

            Q_plot=calcul_Q_all_variable_a2a(nbr_rep,q_min,q_max,V_joint,acc_joint,Tech)
            Q_plot_80=calcul_Q_all_variable_a2a(nbr_rep,q_min,q_max,V_joint*0.8,acc_joint,Tech)
            Q_plot_60=calcul_Q_all_variable_a2a(nbr_rep,q_min,q_max,V_joint*0.6,acc_joint,Tech)
            Q_plot_40=calcul_Q_all_variable_a2a(nbr_rep,q_min,q_max,V_joint*0.4,acc_joint,Tech)
            Q_plot_20=calcul_Q_all_variable_a2a(nbr_rep,q_min,q_max,V_joint*0.2,acc_joint,Tech)

            Q_plot_pourcentage=calcul_Q_all_variable_a2a(nbr_rep,q_min,q_max,V_joint*pourcentage,acc_joint,Tech)
            
            # position vector Q_plot[0]
            # time vector Q_plot[1]
            # velosity vector Q_plot[2]
            # acc vector Q_plot[3]
            time=Q_plot[1]

            # plot the data of the chosen joint

            # print('here the trajectory of joint',joint_i,'the other joints dont move')
            # plot_Trajectory(Q_plot)
            # plot_Trajectory(Q_plot_80)
            # plot_Trajectory(Q_plot_60)
            # plot_Trajectory(Q_plot_40)
            # plot_Trajectory(Q_plot_20)
            
            # calculs of position velosity acceleration for all joint joint with variation 
            # of the Vmax

            Q_total,V_total,A_total=calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot)
            plot_QVA_total(Q_plot[1],nbr_joint,Q_total,V_total,A_total,'max_')

            # Q_total_80,V_total_80,A_total_80=calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot_80)
            # plot_QVA_total(Q_plot_80[1],nbr_joint,Q_total_80,V_total_80,A_total_80,'80_')

            # Q_total_60,V_total_60,A_total_60=calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot_60)
            # plot_QVA_total(Q_plot_60[1],nbr_joint,Q_total_60,V_total_60,A_total_60,'60_')

            # Q_total_40,V_total_40,A_total_40=calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot_40)
            # plot_QVA_total(Q_plot_40[1],nbr_joint,Q_total_40,V_total_40,A_total_40,'40_')

            # Q_total_20,V_total_20,A_total_20=calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot_20)
            # plot_QVA_total(Q_plot_20[1],nbr_joint,Q_total_20,V_total_20,A_total_20,'20_')

            # Q_total_pourcentage,V_total_pourcentage,A_total_pourcentage=calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot_pourcentage)
            # plot_QVA_total(Q_plot_pourcentage[1],nbr_joint,Q_total_pourcentage,V_total_pourcentage,A_total_pourcentage,'%_')

            tau,w=Generate_Torque_Regression_matrix(nbr_joint,Q_total,V_total,A_total)
            phi_etoile=estimation_with_qp_solver(w,tau)
            Generate_text_data_file(Q_total,V_total,A_total,tau)
            
            force=force_coulomb(phi_etoile[21],V_total,nbr_joint)

            plt.figure('force friction')
            for i in range(nbr_joint):
              plt.plot(V_total[i],force[i],linewidth=1, label='fric'+str(i))
            plt.title('friction force')
            plt.xlabel('v')
            plt.ylabel('fric')
            plt.legend()
            plt.show() 

        if(mode==2):

            print('enter the total number of joints')
            nbr_joint=int(input())
            print('enter number of repetition time of motion')
            nbr_rep=int(input())

            time_tau,time_final,q_start,q_end,Vmax,acc_max,D=Data_Alltrajectory(nbr_joint,Tech)
            print('\n')
            print('the new values of velosity and acc after synchronisation')
            for i in range(nbr_joint):
                print('V_joint',i+1,'\t',Vmax[i],'m/s')
                print('acc_joint',i+1,'\t',acc_max[i],'m/s^2')

            for i in range(nbr_joint):

                # q,time,v,a=PreDefined_trajectory(time_tau,time_final,q_start[i],q_end[i], Vmax[i],acc_max[i],D[i],Tech)
                Q_total_per_joint=calcul_Q_all_variable_sync(nbr_rep,time_tau,time_final,q_start[i],q_end[i],Vmax[i],acc_max[i],D[i],Tech)

                Q_total.append(Q_total_per_joint[0])
                V_total.append(Q_total_per_joint[2])
                A_total.append(Q_total_per_joint[3])
                time=Q_total_per_joint[1]

            plot_QVA_total(time,nbr_joint,Q_total,V_total,A_total,'sync')

            tau,w=Generate_Torque_Regression_matrix(nbr_joint,Q_total,V_total,A_total)
            Generate_text_data_file(Q_total,V_total,A_total,tau)


            phi_etoile=estimation_with_qp_solver(w,tau)
            force=force_coulomb(phi_etoile[21],V_total,nbr_joint)
            plt.figure('force friction')
            for i in range(nbr_joint):
              plt.plot(V_total[i],force[i],linewidth=1, label='fric'+str(i))
            plt.title('friction force')
            plt.xlabel('v')
            plt.ylabel('fric')
            plt.legend()
            plt.show() 

            
        else:
            print('please re-enter your choice :)')  


    return Q_total,V_total,A_total,time,force

def Generate_text_data_file_Q_txt(Q_total,V_total,A_total):
    # this function take in input q dq ddq tau for all the joint 
    # and write all the data in a file .txt

    # f = open('/home/fadi/projet_cobot_master2/project_M2/Robot/2DOF/Code/Identification/2dof_data_LC.txt','w')
    package_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
    file_path = package_path + '/src/identification_YASKAWA/All_positiondata_file.txt'
    f = open(file_path,'w')

    nbSamples=np.array(Q_total[0]).size
    q_pin=np.array(Q_total)
    dq_pin=np.array(V_total)
    ddq_pin=np.array(A_total)
    print('shape of Q ',q_pin.shape)

    i=0
    line=[str('q1'),'\t',str('q2'),'\t',str('q3'),'\t',str('q4'),'\t',str('q5'),
                '\t',str('q6')]
    f.writelines(line)
    f.write('\n')
    
    for i in range(nbSamples):
        line=[str(q_pin[0][i]),'\t',str(q_pin[1][i]),'\t',str(q_pin[2][i]),'\t',str(q_pin[3][i]),'\t',str(q_pin[4][i]),
                '\t',str(q_pin[5][i]) ,'\t',str(dq_pin[0][i]),'\t',str(dq_pin[1][i]),'\t',str(dq_pin[2][i]),'\t',str(dq_pin[3][i]),'\t',str(dq_pin[4][i]),
                '\t',str(dq_pin[5][i]) ,'\t',str(ddq_pin[0][i]),'\t',str(ddq_pin[1][i]),'\t',str(ddq_pin[2][i]),'\t',str(ddq_pin[3][i]),'\t',str(ddq_pin[4][i]),
                '\t',str(ddq_pin[5][i])]
        f.writelines(line)
        f.write('\n')
        
    f.close()

def read_tau_q_dq_ddq_fromTxt(nbr_of_joint):

    package_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
    
    # file_path_pos = package_path + '/src/identification_YASKAWA/position yaskawa.txt'
    # file_path_V = package_path + '/src/identification_YASKAWA/velosity yaskawa.txt'
    # file_path_torque = package_path + '/src/identification_YASKAWA/torque_yaskawa.txt '

    # file_path = package_path + '/src/identification_YASKAWA/data_all_2_one_by_one.txt'

    file_path = package_path + '/src/identification_YASKAWA/cuting data test.txt'# sandelle one by one <3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3
    
    # file_path = package_path + '/src/identification_YASKAWA/data_static_5_sec.txt'
    # file_path = package_path + '/src/identification_YASKAWA/Yaskawa max min.txt'
    # file_path = package_path + '/src/identification_YASKAWA/cuting data just palier.txt'
    
    # file_path = package_path + '/src/identification_YASKAWA/data_static.txt'# un palier a 0.2 

    
    # f_pos = open(file_path_pos,'r')
    # f_V = open(file_path_V,'r')
    # f_torque = open(file_path_torque,'r')
    tau_par_ordre=[]
    f=open(file_path,'r')
    # tau = [float(i) for i in f_torque.split()]
    tau1=[]
    tau2=[]
    tau3=[]
    tau4=[]
    tau5=[]
    tau6=[]
    
    q1=[]
    q2=[]
    q3=[]
    q4=[]
    q5=[]
    q6=[]
    q=[]
    
    dq1=[]
    dq2=[]
    dq3=[]
    dq4=[]
    dq5=[]
    dq6=[]
    dq=[]
    
    tau_simu_gazebo=[]

    for line in f:

        data_split = line.strip().split()
        print(data_split)
        
        q1.append(data_split[0])
        q2.append(data_split[1])
        q3.append(data_split[2])
        q4.append(data_split[3])
        q5.append(data_split[4])
        q6.append(data_split[5])
    # for line in f_V:
    #     data_split = line.strip().split('\t')
        
        # dq1.append(data_split[6])
        # dq2.append(data_split[7])
        # dq3.append(data_split[8])
        # dq4.append(data_split[9])
        # dq5.append(data_split[10])
        # dq6.append(data_split[11])

        # tau1.append(data_split[6])
        # tau2.append(data_split[7])
        # tau3.append(data_split[8])
        # tau4.append(data_split[9])
        # tau5.append(data_split[10])
        # tau6.append(data_split[11])

    # for line in f_V:
    #     data_split = line.strip().split('\t')   

        tau1.append(data_split[12])
        tau2.append(data_split[13])
        tau3.append(data_split[14])
        tau4.append(data_split[15])
        tau5.append(data_split[16])
        tau6.append(data_split[17])

        # tau_simu_gazebo.append(data_split[12])
        # tau_simu_gazebo.append(data_split[13])
        # tau_simu_gazebo.append(data_split[14])
        # tau_simu_gazebo.append(data_split[15])
        # tau_simu_gazebo.append(data_split[16])
        # tau_simu_gazebo.append(data_split[17])
    
    
    f.close()
    q.append(q1)
    q.append(q2)
    q.append(q3)
    q.append(q4)
    q.append(q5)
    q.append(q6)
    q=np.array(q)
    q=np.double(q)
    
    # dq.append(dq1)
    # dq.append(dq2)
    # dq.append(dq3)
    # dq.append(dq4)
    # dq.append(dq5)
    # dq.append(dq6)
    # dq=np.array(dq)
    # dq=np.double(dq)

    tau_par_ordre.extend(tau1)
    tau_par_ordre.extend(tau2)
    tau_par_ordre.extend(tau3)
    tau_par_ordre.extend(tau4)
    tau_par_ordre.extend(tau5)
    tau_par_ordre.extend(tau6)
    tau_par_ordre=np.array(tau_par_ordre)
    tau_par_ordre=np.double(tau_par_ordre)

    ddq=[[],[],[],[],[],[]]
    dq_th=[[],[],[],[],[],[]]

    for joint_index in range(nbr_of_joint):

        for i in range(q[0].size-1):
            j=i+1
            dv=(q[joint_index][j]-q[joint_index][i])/Tech
            # print('da=\t','dv',(v[j]-v[i]),'/dt',(time[j]-time[i]),'=',da)
            dq_th[joint_index].append(dv)
        
        dq_th[joint_index].append(dv)
    
    dq_th=np.array(dq_th)
    
    for joint_index in range(nbr_of_joint):

        for i in range(dq_th[0].size-1):
            j=i+1
            da=(dq_th[joint_index][j]-dq_th[joint_index][i])/Tech
            # print('da=\t','dv',(v[j]-v[i]),'/dt',(time[j]-time[i]),'=',da)
            ddq[joint_index].append(da)
        
        ddq[joint_index].append(0)
   
    ddq=np.array(ddq)

    tau_par_ordre=filter_butterworth(int (1/Tech),2,tau_par_ordre)

    for i in range(6):
       q[i]=filter_butterworth(int (1/Tech),2,q[i])
       dq_th[i]=filter_butterworth(int (1/Tech),2,dq_th[i])
       ddq[i]=filter_butterworth(int (1/Tech),2,ddq[i])
    print("shape of q",q.shape)
    
    q=np.array(q)
    dq=np.array(dq)
    ddq=np.array(ddq)
    tau_par_ordre=np.array(tau_par_ordre)
    q=q.T
    dq=dq.T
    ddq=ddq.T
    dq_th=dq_th.T

    return q,dq_th,ddq,tau_par_ordre

def plot_QVA_total(time,nbr_joint,Q_total,V_total,A_total,name):
    # # this function take in input: position of the joint qi
    #                                velosity of the joint dqi
    #                                acceleration of the joint ddqi
    # the function dont return any thing 
    # it plot: the trajectorys of all the joints
            #  the velositys of all joints
            #  the acceleration of all joints
    samples=[]
    for j in range(Q_total[0].size):
        samples.append(j)
    samples=np.array(samples)
    plt.figure('Q_total Trajectory')
    for i in range(nbr_joint):
        plt.plot(samples,Q_total[i],linewidth=1, label='q'+str(name)+str(i))
    plt.title('q Trajectory')
    plt.xlabel('samples')
    plt.ylabel('q(rad)')
    plt.legend()
    plt.show()        

    plt.figure('V_total velocity')
    for i in range(nbr_joint):
        plt.plot(samples,np.array(V_total[i]),linewidth=1, label='V'+str(name)+str(i))
    plt.title('V velocity')
    plt.xlabel('sampels')
    plt.ylabel('V rad/sec')
    plt.legend()
    plt.show() 

    plt.figure('A_total acceleration')
    for i in range(nbr_joint):
        plt.plot(samples,A_total[i],linewidth=1, label='acc'+str(name)+str(i))
    plt.title('acc acceleration')
    plt.xlabel('sampels')
    plt.ylabel('acc rad/s^2')
    plt.legend()
    plt.show() 

def calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot,appending_vect):
    # this function take in input : number of joints 
    #                               a specifique joint joint_i
    #                               and Q_plot the data of joint_i
    # it return the data of all joint in mode axe to axe so the non chosen joint will have:
    #                                                                     q=appending_vect,dq=0,ddq=0  
    #
    joint_3=2
    # joint3=3.141592653589793
    joint3=appending_vect[2]
    Q_total=[]
    V_total=[]
    A_total=[]
    tmp=[]
    # panding zeros for all non moving joints 
    # for i in range(nbr_joint):
    #             if(i==joint_i):
    #                 Q_total.append(Q_plot[0])
    #             else:
    #                 tmp=[]
    #                 for i in range(np.array(Q_plot[0]).size):
    #                     tmp.append(0)
    #                 Q_total.append(tmp)
    
    for i in range(nbr_joint):
        # if(i==joint_i):
        #     # dans tous les cas joint desire fait son truc 
        if(i==joint_3 and joint_3!=joint_i):
            print('je suis dans joint 3 mais pas desire')
            tmp=[]
            for i in range(np.array(Q_plot[0]).size):
                tmp.append(joint3)
            Q_total.append(tmp)
        
        elif((i==joint_i and joint_i!=joint_3)or(i==joint_3 and joint_3==joint_i)):
            print('je suis dans joint 3 desire ou joint desire qui est pas 3')
            Q_total.append(Q_plot[0])

        else:
            tmp=[]
            for j in range(np.array(Q_plot[0]).size):
                tmp.append(appending_vect[i])
            Q_total.append(tmp)
    print()
    for i in range(nbr_joint):
                if(i==joint_i):
                    V_total.append(Q_plot[2])
                else:
                    tmp=[]
                    for i in range(np.array(Q_plot[2]).size):
                        tmp.append(0)
                    V_total.append(tmp)

    for i in range(nbr_joint):
                if(i==joint_i):
                    A_total.append(Q_plot[3])
                else:
                    tmp=[]
                    for i in range(np.array(Q_plot[3]).size):
                        tmp.append(0)
                    A_total.append(tmp)

    # Q_total.append(V_total)
    # Q_total.append(A_total)

    return Q_total,V_total,A_total

def plot_Trajectory(Q):
    # this function plot a pre-calculated trajectory 
    # so it plot q dq ddq in 3 different figure 
    q=Q[0],
    time=Q[1]
    v=Q[2]
    a=Q[3]
    print('shape of Q[0]',np.array(Q[0]).shape)
    print('shape of Q[1]',np.array(Q[1]).shape)
    plt.figure('q Trajectory')
    plt.plot(Q[1],Q[0],linewidth=1, label='q')
    plt.title('q Trajectory')
    plt.xlabel('t')
    plt.ylabel('q')
    plt.legend()
    plt.show()

    plt.figure('V velocity calculated via derivation(dq)')
    plt.title('V velocity calculated via derivation(dq)')
    plt.plot(time,abs(np.array(v)),linewidth=1, label='V')
    plt.xlabel('t sec')
    plt.ylabel('V(m/s)')
    plt.legend()
    plt.show()
    

    plt.figure('acc acceleration calculated via derivation(ddq)')
    plt.plot(time,a,linewidth=1, label='acc')
    plt.xlabel('t sec')
    plt.ylabel('acc (m/s^2)')
    plt.legend()
    plt.show()
    plt.title('acc acceleration calculated via derivation(ddq)')

def calcul_Q_all_variable_sync(nbr_rep,time1,timeEnd,q_min,q_max,V_joint,acc_joint,D,Tech):
    # this function take in input :1- the number of repetition 
    #                              2- the data of each joint motion(qstart qend V acc )

    # And return a matrix that combine:1- position vector after  repetion 
    #                                  2- velosity vector after  repetion 
    #                                  3- acc vector after  repetion 
    #                                  4- time vector with after repetion     

    Q_plot=[]
    Q=[]
    V=[]
    T=[]
    tf=0
    A=[]
    # print('avant nbr_rep')
    for i in range(nbr_rep):
        # print('avant trajectory')      
        q,t,v,a=PreDefined_trajectory(time1,timeEnd,q_min,q_max,V_joint,acc_joint,D,Tech)
        # print('apres trajectory')  
        Q.extend(q)
        V.extend(v)
        A.extend(a)
        for i in range(np.array(t).size):
            t[i]+=tf

        T.extend(t)       
        tf=T[np.array(T).size-1]

        q1,t1,v1,a1=trajectory(q_max,q_min,V_joint,acc_joint,Tech)
        Q.extend(q1)
        V.extend(v1)
        A.extend(a1)

        for i in range(np.array(t1).size):
            t1[i]+=tf

        T.extend(t1)
        tf=T[np.array(T).size-1]

    Q_plot.append(Q)
    Q_plot.append(T)
    Q_plot.append(V)
    Q_plot.append(A)

    return Q_plot

def calcul_Q_all_variable_a2a(nbr_rep,q_min,q_max,V_joint,acc_joint,Tech):
     #this function take in input :1- the number of repetition 
    #                             2- the data of the chosen joint motion(qstart qend V acc )
    # and return the data of the chosen joint in a matrix that combine:
    #                                                       1- position vector after  repetion 
    #                                                       2- velosity vector after  repetion 
    #                                                       3- acc vector after  repetion 
    #                                                       4- time vector with after repetion     

    Q_plot=[]
    Q=[]
    V=[]
    T=[]
    tf=0
    A=[]
    # print('avant nbr_rep')
    for i in range(nbr_rep):
        # print('avant trajectory')      
        q,t,v,a=trajectory(q_min,q_max,V_joint,acc_joint,Tech)
        # print('apres trajectory')  
        Q.extend(q)
        V.extend(v)
        A.extend(a)
        for i in range(np.array(t).size):
            t[i]+=tf

        T.extend(t)       
        tf=T[np.array(T).size-1]

        q1,t1,v1,a1=trajectory(q_max,q_min,V_joint,acc_joint,Tech)
        Q.extend(q1)
        V.extend(v1)
        A.extend(a1)

        for i in range(np.array(t1).size):
            t1[i]+=tf

        T.extend(t1)
        tf=T[np.array(T).size-1]

    Q_plot.append(Q)
    Q_plot.append(T)
    Q_plot.append(V)
    Q_plot.append(A)

    return Q_plot

def trajectory(q_start,q_end,Vmax,acc_max,Tech):

    #function that take as input:
    #                            1-initial position
    #                            2-final position
    #                            3-Max velocity
    #                            4-acceleration
    #                            5- periode of sampling
    #  
    # AND return:
    #            1- trajectory q
    #            2- velocity (dq/dt)
    #            3- acceleration
    #            4- time vector that take the movement to go from qi to qf
     

                                #the max velocity is given by vmax=sqrt(D*ACC)

    D=q_end-q_start             # total distance 
    if(abs(D)>((Vmax*Vmax)/acc_max)):# D>Vmax^2/acc it's condition so the movment can be achivebel(realisable)
        time1=Vmax/acc_max      #time that take the velocity to go from 0 to max
    else:
        print('we need a better value of D')
                                # if D dont respect the condition we need another D 
                                # so another input for the function

    timeEnd=time1+(abs(D)/Vmax)      # time wen the mvt end
    print('Vmax \t acc_max \t time1 \t timeEnd')
    print(Vmax,'\t',acc_max,'\t',time1,'\t',timeEnd)
    t=0
    time=[]
    time.append(t)
    q=[]
    v=[]
    a=[]
    #calculation of q in each intervel of time 
    if(t<=time1):
        q.append(q_start+0.5*t*t*acc_max*sign(D))
        v.append(t*acc_max*sign(D))
        a.append(acc_max*sign(D))
    elif(t<=(timeEnd-time1)):
        q.append(q_start+(t-(time1/2))*Vmax*sign(D))
        v.append(Vmax*sign(D))
        a.append(0)
    elif(t<=timeEnd):
        q.append(q_end-0.5*(timeEnd-t)*(timeEnd-t)*acc_max*sign(D))
        v.append((timeEnd-t)*acc_max*sign(D))
        a.append(-acc_max*sign(D))
    else:
        print('time out of bound')
    
    while(abs(t-timeEnd)>0.01):
        if(t<=time1):
            q.append(q_start+0.5*t*t*acc_max*sign(D))
            v.append(t*acc_max*sign(D))
            a.append(acc_max*sign(D))
        elif(t<=(timeEnd-time1)):
            q.append(q_start+(t-(time1/2))*Vmax*sign(D))
            v.append(Vmax*sign(D))
            a.append(0)
        elif(t<=timeEnd):
            q.append(q_end-0.5*(timeEnd-t)*(timeEnd-t)*acc_max*sign(D))
            v.append((timeEnd-t)*acc_max*sign(D))
            a.append(-acc_max*sign(D))
        else:
            print('time out of bound')
        
        t=t+Tech
        time.append(t)

    # #calculation of the velocity 
    # for i in range(np.array(time).size-1):
    #     j=i+1
    #     dv=(q[j]-q[i])/(time[j]-time[i])
    #     # print('dv=\t',dv)
    #     v.append(dv)
    # v.append(dv)
    # v=np.array(v)
    # vreturn=v
    # v=abs(v)
    # print('shape of time',np.array(time).shape)
    # print('shape of v',np.array(v).shape)
    
    # #calculation of acceleration
    
    # for i in range(np.array(v).size-1):
    #     j=i+1
    #     da=(v[j]-v[i])/(time[j]-time[i])
    #     # print('da=\t','dv',(v[j]-v[i]),'/dt',(time[j]-time[i]),'=',da)
    #     a.append(da)
    # a.append(0)
    # print('shape of time',np.array(time).shape)
    # print('shape of a',np.array(a).shape)

    return q,time,v,a

def Data_Alltrajectory(Nmbrofjoint,Tech):

    # this function take the number of joint and the periode of sampling
    # And return -the first time to go from 0 to max velocity for all joint 
    #            -the final time for all joint to complet the hall trajectory
    #            -the start position q_start
    #            -the final position q_end
    #            -the new joint velocity lamda*Vmax
    #            -the new joint acceleration mu*acc
    #            -the Distance between initial and final position
                  
    q_start=[]
    q_end=[] 
    Vmax=[]
    acc_max=[]
    D=[]
    # taking data for each joint from the user
    for i in range (Nmbrofjoint):
        print('enter joint',i+1,'start position')
        start=float(input())
        q_start.append(start)
        print('enter joint',i+1,'final position')
        end=float(input())
        q_end.append(end)
        print('enter joint',i+1,'velocity')
        Vm=float(input())
        Vmax.append(Vm)
        print('enter joint',i+1,'acceleration')
        am=float(input())
        acc_max.append(am)
    
    #Distance calculation
    for i in range (Nmbrofjoint):
        d=q_end[i]-q_start[i]
        D.append(d)             
    
    # first coefficient calculation lamda_1 and mu_1
    lamda_1=1
    mu_1=1
    lamda_j=[]
    mu_j=[]
    lamda_j.append(lamda_1)
    mu_j.append(mu_1)
    
    for j in range (1,Nmbrofjoint):
        
        lamda=(Vmax[j]*abs(D[0]))/Vmax[0]*abs(D[j])
        mu=(acc_max[j]*abs(D[0]))/acc_max[0]*abs(D[j])
        lamda_j.append(lamda)
        mu_j.append(mu)

    lamda_j=np.array(lamda_j)
    mu_j=np.array(mu_j)
    lamda_1=np.min(lamda_j)
    mu_1=np.min(mu_j)
    

    # calculation of joints coefficients (lamda for each joint and mu for each joint)  
    lamda_joint=[]
    lamda_joint.append(lamda_1)
    for joint in range(1,Nmbrofjoint):
        lamda_joint.append(lamda_1*((Vmax[0]*abs(D[joint]))/(Vmax[joint]*abs(D[0]))))

    mu_joint=[]
    mu_joint.append(mu_1)

    for joint in range(1,Nmbrofjoint):
        mu_joint.append(mu_1*((acc_max[0]*abs(D[joint]))/(acc_max[joint]*abs(D[0]))))

    #Display of coefficient values
    print('LAMDA_J values\n')
    print(lamda_joint)
    print('lamda_1=\t',lamda_1)
    print('MU_J values\n')
    print(mu_joint)
    print('mu_1=\t',mu_1)
    
    # new velocity calculation
    new_Vmax=[]
    NewVmax=lamda_1*Vmax[0]
    new_Vmax.append(NewVmax)
    for j in range(1,Nmbrofjoint):
        new_Vmax.append(lamda_joint[j]*Vmax[j])

    # new acceleration calculation
    new_Amax=[]
    NewAmax=mu_1*acc_max[0]
    new_Amax.append(NewAmax)
    for j in range(1,Nmbrofjoint):
        new_Amax.append(mu_joint[j]*acc_max[j])

    # first time calculation
    time_tau=(lamda_1*Vmax[0])/(mu_1*acc_max[0])

    #final time calculation
    timeEnd=[]
    tf1=((lamda_1*Vmax[0])/(mu_1*acc_max[0]))+(abs(D[0])/(lamda_1*Vmax[0]))
    timeEnd.append(tf1)
    for k in range(1,Nmbrofjoint):     
        if(abs(D[k])>((Vmax[k]*Vmax[k])/acc_max[k])):# D>Vmax^2/acc it's condition so the movment can be achivebel(realisable)
            tf=((lamda_j[k]*Vmax[k])/(mu_j[k]*acc_max[k]))+(D[k]/(lamda_j[k]*Vmax[k]))
            timeEnd.append(tf)                     #time that take the velocity to go from 0 to max
        else:
            print('we need a better value of D for the joint',k+1)
                                # if D dont respect the condition we need another D 
                                # so another input for the function

              # time wen the mvt end
    timeEnd=np.array(timeEnd)
    time_final=np.max(timeEnd)

    return time_tau,time_final,q_start,q_end,new_Vmax,new_Amax,D
    
def PreDefined_trajectory(time1,timeEnd,q_start,q_end, Vmax,acc_max,D,Tech):
    # this function is dedicated for the calculus of a trajectory with predefine data
    # the first time and final time are already calculated 
    # this function return :-position vector q
    #                       -time vector time
    #                       -velocity vector v
    #                       -acceleration vector a

    t=0
    time=[]
    time.append(t)
    q=[]
    v=[]
    a=[]
    #calculation of q in each intervel of time 
    if(t<=time1):
        q.append(q_start+0.5*t*t*acc_max*sign(D))
    elif(t<=(timeEnd-time1)):
        q.append(q_start+(t-(time1/2))*Vmax*sign(D))
    elif(t<=timeEnd):
        q.append(q_end-0.5*(timeEnd-t)*(timeEnd-t)*acc_max*sign(D))
    else:
        print('time out of bound')
    
    while(abs(t-timeEnd)>0.01):
        if(t<=time1):
            q.append(q_start+0.5*t*t*acc_max*sign(D))
        elif(t<=(timeEnd-time1)):
            q.append(q_start+(t-(time1/2))*Vmax*sign(D))
        elif(t<=timeEnd):
            q.append(q_end-0.5*(timeEnd-t)*(timeEnd-t)*acc_max*sign(D))
        else:
            print('time out of bound')
        
        t=t+Tech
        time.append(t)
        
        #calculation of the velocity 
    for i in range(np.array(time).size-1):
        j=i+1
        dv=(q[j]-q[i])/(time[j]-time[i])
        # print('dv=\t',dv)
        v.append(dv)
    v.append(dv)
    v=np.array(v)
    vreturn=v
    v=abs(v)
    print('shape of time',np.array(time).shape)
    print('shape of v',np.array(v).shape)
    
        #calculation of acceleration
    
    for i in range(np.array(v).size-1):
        j=i+1
        da=(v[j]-v[i])/(time[j]-time[i])
        # print('da=\t','dv',(v[j]-v[i]),'/dt',(time[j]-time[i]),'=',da)
        a.append(da)
    a.append(0)
    print('shape of time',np.array(time).shape)
    print('shape of a',np.array(a).shape)

    return q,time,vreturn,a

def PreDefined_velocity_trajectory(time1,timeEnd,Vmax,acc_max,Tech):
        # this function calculate the evolution of velocity  wen time evolute
        # it's based on the law of movement calculation
        # the purpose of this function is to see the theoraticale evolution of velocity 
        # this function return velocity and time vectors
    t=0
    time=[]
    time.append(t)
    q=[]
    v=[]
    a=[]
    #calculation of Velocity in each intervel of time 
    if(t<=time1):
        v.append(acc_max*t)
    elif(t<=(timeEnd-time1)):
        v.append(acc_max*time1)   
    elif(t<=timeEnd):
         v.append(acc_max*(timeEnd-t))
    else:
        print('time out of bound')
    
    while(abs(t-timeEnd)>0.01):
        if(t<=time1):
            v.append(acc_max*t)
        elif(t<=(timeEnd-time1)):
            v.append(acc_max*time1)
        elif(t<=timeEnd):
             v.append(acc_max*(timeEnd-t))
        else:
            print('time out of bound')
        
        t=t+Tech
        time.append(t)
    return v,time

def Generate_Torque_Regression_matrix(nbr_joint,Q_total,V_total,A_total):

    # esopilome dont use this function
    #this function take as input number of joints and the data of each joint (q dq ddq)
    #it return the torque of each joint and the regression matrix

    # Generate ouput with pin
    nbSamples=np.array(Q_total[0]).size
    tau=[]
    q=np.array(Q_total)
    dq=np.array(V_total)
    ddq=np.array(A_total)
    print('shape of q test1 \t','\t',q.shape)

    for i in range(nbSamples):
        tau.extend(pin.rnea(model, data, q[:, i], dq[:, i], ddq[:, i]))
    # print('Shape of tau_pin:\t', np.array(tau).shape)
    tau=np.array(tau)
    tau=np.double(tau)
    print ('shape of tau in the function generate output',tau.shape)

    # # ========== Step 4 - Create IDM with pinocchio (regression matrix)
    w = [] # Regression vector
        ## w pour I/O generer par pinocchio
    for i in range(nbSamples):
        w.extend(pin.computeJointTorqueRegressor(model, data, q[:, i], dq[:, i], ddq[:, i]))
    w=np.array(w)

    print('shape of w********************************************************',w.shape)

    # to add the friction parameters we have to add to the regression matrix a velosity vector
    #  for each joint so we add 2 new vector for each joint in the regression matrix do we will identify 
    # two new friction parameter per joint
    
    #joint 0  
    Z=np.zeros((5*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(dq[0])#adding joint 1 velosity vector
    dq_stack_pin.extend(Z)# panding eith zeros
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w=np.concatenate([w,dq_stack_pin], axis=1)# adding vector to the regressor
    w=np.concatenate([w,dq_sign_pin], axis=1)# adding vector to the regressor
   
    
    
    #joint 1  
    Z1=np.zeros((1*nbSamples,1))
    Z2=np.zeros((4*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(Z1)# adding zeros
    dq_stack_pin.extend(dq[1])#adding joint 1 velosity vector
    dq_stack_pin.extend(Z2)# panding with zeros
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w=np.concatenate([w,dq_stack_pin], axis=1)# adding vector to the regressor
    w=np.concatenate([w,dq_sign_pin], axis=1)# adding vector to the regressor
    
    #joint 2  
    Z1=np.zeros((2*nbSamples,1))
    Z2=np.zeros((3*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(Z1)# adding zeros
    dq_stack_pin.extend(dq[2])#adding joint 1 velosity vector
    dq_stack_pin.extend(Z2)# panding with zeros
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w=np.concatenate([w,dq_stack_pin], axis=1)# adding vector to the regressor
    w=np.concatenate([w,dq_sign_pin], axis=1)# adding vector to the regressor

    #joint 3  
    Z1=np.zeros((3*nbSamples,1))
    Z2=np.zeros((2*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(Z1)# adding zeros
    dq_stack_pin.extend(dq[3])#adding joint 1 velosity vector
    dq_stack_pin.extend(Z2)# panding with zeros
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w=np.concatenate([w,dq_stack_pin], axis=1)# adding vector to the regressor
    w=np.concatenate([w,dq_sign_pin], axis=1)# adding vector to the regressor

    #joint 4
    Z1=np.zeros((4*nbSamples,1))
    Z2=np.zeros((1*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(Z1)# adding zeros
    dq_stack_pin.extend(dq[4])#adding joint 1 velosity vector
    dq_stack_pin.extend(Z2)# panding with zeros
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w=np.concatenate([w,dq_stack_pin], axis=1)# adding vector to the regressor
    w=np.concatenate([w,dq_sign_pin], axis=1)# adding vector to the regressor

    #joint 5
    Z1=np.zeros((5*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(Z1)# adding zeros
    dq_stack_pin.extend(dq[5])#adding joint 1 velosity vector
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w=np.concatenate([w,dq_stack_pin], axis=1)# adding vector to the regressor
    w=np.concatenate([w,dq_sign_pin], axis=1)# adding vector to the regressor

    
    
    #Display of shapes
    print('Shape of w finale avec coco l\'amour:\t',np.array(w).shape)
   

    return tau,w

def estimation_with_qp_solver(w,tau):
    # this function take in input the regression matrix and the torque vectore of the joint
    # it return the estimeted parameters 
    P = np.dot(w.transpose(),w)
    q = -np.dot(tau.transpose(),w)

    # test if P is positive-definite if not then p=spd (spd=symmetric positive semidefinite)
    P=nearestPD(P)

    # phi_etoile=qpsolvers.solve_ls(P,q,None,None)
    phi_etoile=qpsolvers.solve_qp(
            P,
            q,
            G=None,#G Linear inequality matrix.
            h=None,#Linear inequality vector.
            A=None,
            b=None,
            lb=None,
            ub=None,
            solver="quadprog",
            initvals=None,
            sym_proj=True
            )

    phi_etoile=np.array(phi_etoile)
    print("shape of phi_etoile",phi_etoile.shape)
    # phi_etoile=float(phi_etoile)

    tau_estime=np.dot(w,phi_etoile)

    samples = []
    for i in range(np.array(tau).size):
        samples.append(i)

    plt.figure('torque et torque estime')
    plt.plot(samples, tau, 'g', linewidth=1, label='tau')
    plt.plot(samples,tau_estime, 'b', linewidth=0.5, label='tau estime')
    # plt.plot(samples, tau_estime1, 'r', linewidth=1, label='tau estime 1')
    plt.title('tau and tau_estime')
    plt.xlabel(' Samples')
    plt.ylabel('Torque (N/m)')
    plt.legend()
    plt.show()

    err = []
    for i in range(np.array(tau).size):
        err.append(abs(tau[i] - tau_estime[i]) * abs(tau[i] - tau_estime[i]))
    plt.plot(samples, err, linewidth=2, label="err")
    plt.title("erreur quadratique")
    plt.legend()
    plt.show()



    return phi_etoile,tau_estime

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd 

    spd=symmetric positive semidefinite

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    B=np.matrix(B,dtype='float')
    _, s, V = np.linalg.svd(B)#provides another way to factorize a matrix, into singular vectors and singular values

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))#Return the distance between norm(A) and the nearest adjacent number
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        #Cholesky's method serves a test of positive definiteness
        # The Cholesky decomposition (or the Cholesky factorization) 
        # is the factorization of a matrix A into the product of a lower triangular matrix L and its transpose. 
        # We can rewrite this decomposition in mathematical notation as: A = L·LT .
        return True
    except np.linalg.LinAlgError:
        return False

def force_coulomb(FS,V_total,nbr_joint):
    # this function take as input : number of joints
    #                               Velosity of joints
    #                               the friction parametres 
    # it return: the coulombs force vector for each joint 
    Force=[]
    V_total=np.array(V_total)
    for j in range(nbr_joint):
        F=[]
        for i in range(V_total[j].size):
            f=FS*np.sign(V_total[j,i])
            # f=FS*(V_total[j,i])
            f=double(f)
            F.extend([f])

        Force.append(F)   

    print(np.array(Force).shape)

    # Force contient les force de toutes les articulation
    return(Force)

def generateQuinticPolyTraj(Jc0,Jcf,model,param):

    # depreciated
    q=np.zeros(param['NbSample_interpolate'])
    dq=np.zeros(param['NbSample_interpolate'])
    ddq=np.zeros(param['NbSample_interpolate'])

    tf=param['NbSample_interpolate']*param['Ts']

    a=np.zeros(6)
    a[0]=Jc0[0]
    a[1]=Jc0[1]
    a[2]=Jc0[2]/2
    a[3]=( 20*Jcf[0]-20*Jc0[0] -(8*Jcf[1]+12*Jc0[1])*tf -(3*Jc0[2]-Jcf[2])*tf**2 )/(2*tf**3)
    a[4]=( 30*Jc0[0]-30*Jcf[0] +(14*Jcf[1]+16*Jc0[1])*tf +(3*Jc0[2]-2*Jcf[2])*tf**2 )/(2*tf**4)
    a[5]=( 12*Jcf[0]-12*Jc0[0] -(6*Jcf[1]+6*Jc0[1])*tf -(Jc0[2]-Jcf[2])*tf**2 )/(2*tf**5)

    t=0
    for i in range( param['NbSample_interpolate'] ):
        t=t+param['Ts']
        
        q[i]=a[0]+a[1]*t +a[2]*t**2 +a[3]*t**3   +a[4]*t**4      +a[5]*t**5
        dq[i]=    a[1]   +2*a[2]*t  +3*a[3]*t**2 +4*a[4]*t**3    +5*a[5]*t**4
        ddq[i]=          +2*a[2]    +6*a[3]*t    +12*a[4]*t**2    +20*a[5]*t**3       
    return q, dq ,ddq

def generateQuinticPolyTraj_version_GF(Jc0,Jcf,vmax,Tech):
    
    T=[]
    vmax=vmax
    
    tf=15*np.abs(Jcf[0]-Jc0[0])/(8*vmax)
    NbSample_interpolate=int (tf/Tech) +1

    q=np.zeros(NbSample_interpolate)
    dq=np.zeros(NbSample_interpolate)
    ddq=np.zeros(NbSample_interpolate)

    A=np.array([[1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0 ],
            [1, tf, tf**2, tf**3, tf**4, tf**5],
            [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
            [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]])



    a=np.matmul(np.linalg.inv(A),np.hstack((Jc0,Jcf)))


    t=0
    for i in range(NbSample_interpolate):
    
        q[i]=a[0]+a[1]*t +a[2]*t**2 +a[3]*t**3   +a[4]*t**4      +a[5]*t**5
        dq[i]=    a[1]   +2*a[2]*t  +3*a[3]*t**2 +4*a[4]*t**3    +5*a[5]*t**4
        ddq[i]=          +2*a[2]    +6*a[3]*t    +12*a[4]*t**2    +20*a[5]*t**3      
        t=t+Tech

    return q, dq ,ddq,T

def Bang_Bang_acceleration_profile(q_start,q_end,v_max,a_max,Tech):
# depreciated
    q=[]
    dq=[]
    ddq=[]
    D=q_end-q_start
    vect=[(2*abs(D))/(v_max),2*sqrt((abs(D))/(a_max))]
    tf=np.max(np.array(vect))
    t=0
    while(t<=tf):
        if(t<=(tf/2)):
            q_=q_start+2*D*(t/tf)**2
            q.append(q_)
            dq_=((4*D)/(tf*tf))*t
            dq.append(dq_)
            ddq_=(4*D)/(tf*tf)
            ddq.append(ddq_)
        if(t>(tf/2)):
            q_=q_start+(-1+4*(t/tf)-2*(t/tf)**2)*D
            q.append(q_)
            dq_=((-4*D)/(tf*tf))*t+(4*D)/tf
            dq.append(dq_)
            ddq_=(-4*D)/(tf*tf)
            ddq.append(ddq_)
        t=t+Tech

    return q,dq,ddq

def estimated_parameters_from_data_file(nbr_joint,Q_total_All_Joint,V_total_All_Joint,A_total_All_Joint,tau_simu_gazebo):
# depreciated

    tau_pin,w=Generate_Torque_Regression_matrix(nbr_joint,Q_total_All_Joint,V_total_All_Joint,A_total_All_Joint)
    # phi_etoile_pin=estimation_with_qp_solver(w,tau_pin)
    phi_etoile,tau_estime=estimation_with_qp_solver(w,tau_pin)
    
    
    # plt.figure('force friction')
    # for i in range(nbr_joint):
    #     plt.plot(V_total[i],force[i],linewidth=1, label='fric'+str(i))
    # plt.plot(samples,q,linewidth=1, label='fric'+str(i))
    # plt.title('friction force')
    # plt.xlabel('v')
    # plt.ylabel('fric')
    # plt.legend()
    # plt.show() 
    force=force_coulomb(phi_etoile[21],V_total_All_Joint,nbr_joint)
    
    return phi_etoile

def genrate_W_And_torque_simulation_pin(Q_total,V_total,A_total):
# depreciated
    nbSamples=np.array(Q_total[0]).size
    q_pin=np.array(Q_total)
    dq_pin=np.array(V_total)
    ddq_pin=np.array(A_total)

    tau_pin=[]
    # Generate ouput with pin
    for i in range(nbSamples):
        tau_pin.extend(pin.rnea(model, data, q_pin[:, i], dq_pin[:, i], ddq_pin[:, i]))
    print('Shape of tau_pin:\t', np.array(tau_pin).shape)
    
    tau_pin=np.array(tau_pin)
    w_pin=[]

    ## w pour I/O generer par pinocchio

    for i in range(nbSamples):
        w_pin.extend(pin.computeJointTorqueRegressor(model, data, q_pin[:, i], dq_pin[:, i], ddq_pin[:, i]))
    w_pin=np.array(w_pin)
    
    #joint 0  
    Z=np.zeros((5*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(dq_pin[0])#adding joint 1 velosity vector
    dq_stack_pin.extend(Z)# panding eith zeros
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w_pin=np.concatenate([w_pin,dq_stack_pin], axis=1)# adding vector to the regressor
    w_pin=np.concatenate([w_pin,dq_sign_pin], axis=1)# adding vector to the regressor

    #joint 1  
    Z1=np.zeros((1*nbSamples,1))
    Z2=np.zeros((4*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(Z1)# adding zeros
    dq_stack_pin.extend(dq_pin[1])#adding joint 1 velosity vector
    dq_stack_pin.extend(Z2)# panding with zeros
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w_pin=np.concatenate([w_pin,dq_stack_pin], axis=1)# adding vector to the regressor
    w_pin=np.concatenate([w_pin,dq_sign_pin], axis=1)# adding vector to the regressor

    #joint 2  
    Z1=np.zeros((2*nbSamples,1))
    Z2=np.zeros((3*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(Z1)# adding zeros
    dq_stack_pin.extend(dq_pin[2])#adding joint 1 velosity vector
    dq_stack_pin.extend(Z2)# panding with zeros
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w_pin=np.concatenate([w_pin,dq_stack_pin], axis=1)# adding vector to the regressor
    w_pin=np.concatenate([w_pin,dq_sign_pin], axis=1)# adding vector to the regressor

    #joint 3  
    Z1=np.zeros((3*nbSamples,1))
    Z2=np.zeros((2*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(Z1)# adding zeros
    dq_stack_pin.extend(dq_pin[3])#adding joint 1 velosity vector
    dq_stack_pin.extend(Z2)# panding with zeros
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w_pin=np.concatenate([w_pin,dq_stack_pin], axis=1)# adding vector to the regressor
    w_pin=np.concatenate([w_pin,dq_sign_pin], axis=1)# adding vector to the regressor

    #joint 4
    Z1=np.zeros((4*nbSamples,1))
    Z2=np.zeros((1*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(Z1)# adding zeros
    dq_stack_pin.extend(dq_pin[4])#adding joint 1 velosity vector
    dq_stack_pin.extend(Z2)# panding with zeros
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w_pin=np.concatenate([w_pin,dq_stack_pin], axis=1)# adding vector to the regressor
    w_pin=np.concatenate([w_pin,dq_sign_pin], axis=1)# adding vector to the regressor

    #joint 5
    Z1=np.zeros((5*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(Z1)# adding zeros
    dq_stack_pin.extend(dq_pin[5])#adding joint 1 velosity vector
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w_pin=np.concatenate([w_pin,dq_stack_pin], axis=1)# adding vector to the regressor
    w_pin=np.concatenate([w_pin,dq_sign_pin], axis=1)# adding vector to the regressor

    #Display of shapes
    print('Shape of W_pin:\t',w_pin.shape)

    ## calculation of a positive-definite matrix 
    # QP_solver
    w_pin=np.double(w_pin)
    p_pin=np.dot(w_pin.transpose(),w_pin)
    q_pin= -np.dot(tau_pin.transpose(),w_pin)
    p_pin=nearestPD(p_pin)


    G=np.zeros((18,72))
    for i in range(4):
        G[i,i]=-1
        # G[i+4,i+6]=-1
        # G[i+8,i+12]=-1
        # G[i+12,i+18]=-1
        # G[i+16,i+24]=-1
        # G[i+20,i+28]=-1
        
    h=np.zeros((1,4))
    # h=[-5,-5,-5,-5, -5,-5,-5,-5, -5,-5,-5,-5, -5,-5,-5,-5, -5,-5,-5,-5, -5,-5,-5,-5]
  
    G=np.double(G)
    h=np.double(h)

    phi_etoile_pin=qpsolvers.solve_qp(
                p_pin,
                q_pin,
                G=None,
                h=None,
                A=None,
                b=None,
                lb=None,
                ub=None,
                solver="quadprog",
                initvals=None,
                sym_proj=True
                )

    print('*****************************************')
    # print('phi_etoile',phi_etoile.shape)
    phi_etoile_pin=np.array(phi_etoile_pin)
    phi_etoile_pin=np.double(phi_etoile_pin)
    print('phi_etoile_pin',phi_etoile_pin.shape)
    print('phi_etoile_pin_yaskawa value',phi_etoile_pin)
    print('*****************************************')

    tau_estime_pin=np.dot(w_pin,phi_etoile_pin)
    print('shape of tau_estime',tau_estime_pin.shape)

    samples = []
    for i in range(np.array(tau_pin).size):
        samples.append(i)

    err = []
    for i in range(np.array(tau_pin).size):
        err.append(abs(tau_pin[i] - tau_estime_pin[i]) * abs(tau_pin[i] - tau_estime_pin[i]))
    plt.plot(samples, err, linewidth=2, label="err")
    plt.title("erreur quadratique")
    plt.legend()
    plt.show()
    return(phi_etoile_pin)

def genrate_W_And_torque_pin_rand(nbSamples):
# depreciated
    q_pin = np.random.rand(NQ, nbSamples) * np.pi - np.pi/2  # -pi/2 < q < pi/2
    dq_pin = np.random.rand(NQ, nbSamples) * 10              # 0 < dq  < 10
    ddq_pin = np.random.rand(NQ, nbSamples) * 2               # 0 < dq  < 2
    # tau_pin = np.random.rand(NQ*nbSamples) * 4
    
    tau_pin=[]
    # Generate ouput with pin
    for i in range(nbSamples):
        tau_pin.extend(pin.rnea(model, data, q_pin[:, i], dq_pin[:, i], ddq_pin[:, i]))
    print('Shape of tau_pin:\t', np.array(tau_pin).shape)
    
    tau_pin=np.array(tau_pin)
    w_pin=[]

    ## w pour I/O generer par pinocchio

    for i in range(nbSamples):
        w_pin.extend(pin.computeJointTorqueRegressor(model, data, q_pin[:, i], dq_pin[:, i], ddq_pin[:, i]))
    w_pin=np.array(w_pin)
    
    #joint 0  
    Z=np.zeros((5*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(dq_pin[0])#adding joint 1 velosity vector
    dq_stack_pin.extend(Z)# panding eith zeros
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w_pin=np.concatenate([w_pin,dq_stack_pin], axis=1)# adding vector to the regressor
    w_pin=np.concatenate([w_pin,dq_sign_pin], axis=1)# adding vector to the regressor

    #joint 1  
    Z1=np.zeros((1*nbSamples,1))
    Z2=np.zeros((4*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(Z1)# adding zeros
    dq_stack_pin.extend(dq_pin[1])#adding joint 1 velosity vector
    dq_stack_pin.extend(Z2)# panding with zeros
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w_pin=np.concatenate([w_pin,dq_stack_pin], axis=1)# adding vector to the regressor
    w_pin=np.concatenate([w_pin,dq_sign_pin], axis=1)# adding vector to the regressor

    #joint 2  
    Z1=np.zeros((2*nbSamples,1))
    Z2=np.zeros((3*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(Z1)# adding zeros
    dq_stack_pin.extend(dq_pin[2])#adding joint 1 velosity vector
    dq_stack_pin.extend(Z2)# panding with zeros
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w_pin=np.concatenate([w_pin,dq_stack_pin], axis=1)# adding vector to the regressor
    w_pin=np.concatenate([w_pin,dq_sign_pin], axis=1)# adding vector to the regressor

    #joint 3  
    Z1=np.zeros((3*nbSamples,1))
    Z2=np.zeros((2*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(Z1)# adding zeros
    dq_stack_pin.extend(dq_pin[3])#adding joint 1 velosity vector
    dq_stack_pin.extend(Z2)# panding with zeros
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w_pin=np.concatenate([w_pin,dq_stack_pin], axis=1)# adding vector to the regressor
    w_pin=np.concatenate([w_pin,dq_sign_pin], axis=1)# adding vector to the regressor

    #joint 4
    Z1=np.zeros((4*nbSamples,1))
    Z2=np.zeros((1*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(Z1)# adding zeros
    dq_stack_pin.extend(dq_pin[4])#adding joint 1 velosity vector
    dq_stack_pin.extend(Z2)# panding with zeros
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w_pin=np.concatenate([w_pin,dq_stack_pin], axis=1)# adding vector to the regressor
    w_pin=np.concatenate([w_pin,dq_sign_pin], axis=1)# adding vector to the regressor

    #joint 5
    Z1=np.zeros((5*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(Z1)# adding zeros
    dq_stack_pin.extend(dq_pin[5])#adding joint 1 velosity vector
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w_pin=np.concatenate([w_pin,dq_stack_pin], axis=1)# adding vector to the regressor
    w_pin=np.concatenate([w_pin,dq_sign_pin], axis=1)# adding vector to the regressor

    #Display of shapes
    print('Shape of W_pin:\t',w_pin.shape)

    ## calculation of a positive-definite matrix 
    # QP_solver
    w_pin=np.double(w_pin)
    p_pin=np.dot(w_pin.transpose(),w_pin)
    q_pin= -np.dot(tau_pin.transpose(),w_pin)
    p_pin=nearestPD(p_pin)


    phi_etoile_pin=qpsolvers.solve_qp(
                p_pin,
                q_pin,
                G=None,
                h=None,
                A=None,
                b=None,
                lb=None,
                ub=None,
                solver="quadprog",
                initvals=None,
                sym_proj=True
                )

    print('*****************************************')
    # print('phi_etoile',phi_etoile.shape)
    phi_etoile_pin=np.array(phi_etoile_pin)
    phi_etoile_pin=np.double(phi_etoile_pin)
    print('phi_etoile_pin',phi_etoile_pin.shape)
    print('phi_etoile_pin_yaskawa value',phi_etoile_pin)
    print('*****************************************')

    tau_estime_pin=np.dot(w_pin,phi_etoile_pin)
    print('shape of tau_estime',tau_estime_pin.shape)

    samples = []
    for i in range(np.array(tau_pin).size):
        samples.append(i)

    err = []
    for i in range(np.array(tau_pin).size):
        err.append(abs(tau_pin[i] - tau_estime_pin[i]) * abs(tau_pin[i] - tau_estime_pin[i]))
    plt.plot(samples, err, linewidth=2, label="err")
    plt.title("erreur quadratique")
    plt.legend()
    plt.show()

    return(phi_etoile_pin)

def genrate_W_And_torque_experimentale(Q_total,V_total,A_total,tau_experimentale):
# depreciated
    nbSamples=np.array(Q_total[0]).size
    q_pin=np.array(Q_total)
    dq_pin=np.array(V_total)
    ddq_pin=np.array(A_total)
    
    param['NbSample']=int(nbSamples)
    
    
    tau_pin=np.array(tau_experimentale)
    w_pin=[]

    ## w pour I/O generer par pinocchio

    for i in range(nbSamples):
        w_pin.extend(pin.computeJointTorqueRegressor(model, data, q_pin[:, i], dq_pin[:, i], ddq_pin[:, i]))
    w_pin=np.array(w_pin)

    #This function calculates joint torques and generates the joint torque regressor.
    #Note: a parameter Friction as to be set to include in dynamic model
    #Input: model, data: model and data structure of robot from Pinocchio
    #q, v, a: joint's position, velocity, acceleration
    #N : number of samples
    #nq: length of q
    #Output: tau: vector of joint torque
    #W : joint torque regressor"""


    #joint 0  
    Z=np.zeros((5*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(dq_pin[0])#adding joint 1 velosity vector
    dq_stack_pin.extend(Z)# panding eith zeros
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w_pin=np.concatenate([w_pin,dq_stack_pin], axis=1)# adding vector to the regressor
    w_pin=np.concatenate([w_pin,dq_sign_pin], axis=1)# adding vector to the regressor

    #joint 1  
    Z1=np.zeros((1*nbSamples,1))
    Z2=np.zeros((4*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(Z1)# adding zeros
    dq_stack_pin.extend(dq_pin[1])#adding joint 1 velosity vector
    dq_stack_pin.extend(Z2)# panding with zeros
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w_pin=np.concatenate([w_pin,dq_stack_pin], axis=1)# adding vector to the regressor
    w_pin=np.concatenate([w_pin,dq_sign_pin], axis=1)# adding vector to the regressor

    #joint 2  
    Z1=np.zeros((2*nbSamples,1))
    Z2=np.zeros((3*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(Z1)# adding zeros
    dq_stack_pin.extend(dq_pin[2])#adding joint 1 velosity vector
    dq_stack_pin.extend(Z2)# panding with zeros
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w_pin=np.concatenate([w_pin,dq_stack_pin], axis=1)# adding vector to the regressor
    w_pin=np.concatenate([w_pin,dq_sign_pin], axis=1)# adding vector to the regressor

    #joint 3  
    Z1=np.zeros((3*nbSamples,1))
    Z2=np.zeros((2*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(Z1)# adding zeros
    dq_stack_pin.extend(dq_pin[3])#adding joint 1 velosity vector
    dq_stack_pin.extend(Z2)# panding with zeros
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w_pin=np.concatenate([w_pin,dq_stack_pin], axis=1)# adding vector to the regressor
    w_pin=np.concatenate([w_pin,dq_sign_pin], axis=1)# adding vector to the regressor

    #joint 4
    Z1=np.zeros((4*nbSamples,1))
    Z2=np.zeros((1*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(Z1)# adding zeros
    dq_stack_pin.extend(dq_pin[4])#adding joint 1 velosity vector
    dq_stack_pin.extend(Z2)# panding with zeros
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w_pin=np.concatenate([w_pin,dq_stack_pin], axis=1)# adding vector to the regressor
    w_pin=np.concatenate([w_pin,dq_sign_pin], axis=1)# adding vector to the regressor

    #joint 5
    Z1=np.zeros((5*nbSamples,1))
    # print('shape of Z',Z.shape)
    dq_stack_pin=[]
    dq_stack_pin_augmente=[]
    dq_stack_pin.extend(Z1)# adding zeros
    dq_stack_pin.extend(dq_pin[5])#adding joint 1 velosity vector
    dq_stack_pin=np.array([dq_stack_pin])
    dq_stack_pin=dq_stack_pin.T
    dq_sign_pin=np.sign(dq_stack_pin)#adding sing(dq)
    w_pin=np.concatenate([w_pin,dq_stack_pin], axis=1)# adding vector to the regressor
    w_pin=np.concatenate([w_pin,dq_sign_pin], axis=1)# adding vector to the regressor

    #Display of shapes
    print('Shape of W_pin:\t',w_pin.shape)

    ## calculation of a positive-definite matrix 
    # QP_solver
    w_pin=np.double(w_pin)
    p_pin=np.dot(w_pin.transpose(),w_pin)
    q_pin= -np.dot(tau_pin.transpose(),w_pin)
    p_pin=nearestPD(p_pin)

    # G=np.zeros((24,72))
    #constraint masse positive
    # for i in range(4):
    #     G[i,i]=-1
    #     G[i+4,i+6]=-1
    #     G[i+8,i+12]=-1
    #     G[i+12,i+18]=-1
    #     G[i+16,i+24]=-1
    #     G[i+20,i+28]=-1
    G=np.zeros((6,72))
    i=0
    G[i,i]=-1
    G[i+1,i+10]=-1
    G[i+2,i+20]=-1
    G[i+3,i+30]=-1
    G[i+4,i+40]=-1
    G[i+5,i+50]=-1
        
    # h=np.zeros((1,24))
    # h=[-1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1]
    h=[-5,-5,-5,-5,-5,-4]
    G=np.double(G)
    h=np.double(h)
        #constraint masse positive
    # G=([-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #    [0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #    [0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #    [0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0])
    # h=[0,0,0,0,0,0,0,0]

    phi_etoile_pin=qpsolvers.solve_qp(
                p_pin,
                q_pin,
                G,
                h,
                A=None,
                b=None,
                lb=None,
                ub=None,
                solver="quadprog",
                initvals=None,
                sym_proj=True
                )

    print('*****************************************')
    # print('phi_etoile',phi_etoile.shape)
    phi_etoile_pin=np.array(phi_etoile_pin)
    phi_etoile_pin=np.double(phi_etoile_pin)
    print('phi_etoile_pin',phi_etoile_pin.shape)
    print('phi_etoile_pin_yaskawa value',phi_etoile_pin)
    print('*****************************************')

    tau_estime_pin=np.dot(w_pin,phi_etoile_pin)
    print('shape of tau_estime',tau_estime_pin.shape)

    samples = []
    for i in range(np.array(tau_pin).size):
        samples.append(i)

    plt.figure('torque estime et torque mesure par le robot')
    # plt.plot(samples, tau_pin, 'g', linewidth=2, label='tau')
    plt.plot(samples, tau_pin, 'g', linewidth=1, label='tau_robot')
    plt.plot(samples,tau_estime_pin, 'b', linewidth=1, label='tau estime ')
    plt.title('tau robot et tau estime')
    plt.xlabel(' Samples')
    plt.ylabel('parametres')
    plt.legend()
    plt.show()
    
    err = []
    for i in range(np.array(tau_pin).size):
        err.append(abs(tau_pin[i] - tau_estime_pin[i]) * abs(tau_pin[i] - tau_estime_pin[i]))
    plt.plot(samples, err, linewidth=2, label="err")
    plt.title("erreur quadratique")
    plt.legend()
    plt.show()

    return(phi_etoile_pin)

def Base_regressor(Q_total,V_total,A_total,tau_experimentale):
    # depreciated
    nbSamples=np.array(Q_total[0]).size
    q_pin=np.array(Q_total)
    dq_pin=np.array(V_total)
    ddq_pin=np.array(A_total)

    names = []
    for i in range(1, NJOINT):
        names += ['m'+str(i), 'mx'+str(i), 'my'+str(i), 'mz'+str(i), 'Ixx'+str(i),
                'Ixy'+str(i), 'Iyy'+str(i), 'Izx'+str(i), 'Izy'+str(i), 'Izz'+str(i)]

    phi = []
    for i in range(1, NJOINT):
        phi.extend(model.inertias[i].toDynamicParameters())

    print('shape of phi:\t', np.array(phi).shape)

    # ========== Step 3 - Generate input and output - 1000 samples (than's data) 

    tau=tau_experimentale#

    # # ========== Step 4 - Create IDM with pinocchio (regression matrix)
    # w = []  # Regression vector
    
    w_pin=[]
    # w pour I/O generer par pinocchio

    for i in range(nbSamples):
        w_pin.extend(pin.computeJointTorqueRegressor(model, data, q_pin[:, i], dq_pin[:, i], ddq_pin[:, i]))
    w_pin=np.array(w_pin)

    # for i in range(6):
    #     w_pin.extend(pin.computeJointTorqueRegressor(model, data, q_pin[i, :], dq_pin[i,:], ddq_pin[i,:]))
    # w_pin=np.array(w_pin)
    
    # ========== Step 5 - Remove non dynamic effect columns then remove zero value columns then remove the parameters related to zero value columns at the end we will have a matix W_modified et Phi_modified

    threshold = 0.000001
    ## we have just to change the w_modified so we can work with the different input/output
    #  w is calculated with the input/output from than's data file
    #  W_pin is calculated with the input/output generated by pinocchio
    

    # W_modified = np.array(w[:])

    W_modified = np.array(w_pin[:])
    index_param_to_delete=[12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    
    indexQR=48
    
    # tmp = []
    # for i in range(len(phi)):
    #     if (np.dot([W_modified[:, i]], np.transpose([W_modified[:, i]]))[0][0] <= threshold):
    #         tmp.append(i)
    # tmp.sort(reverse=True)

    phi_modified = phi[:]
    names_modified = names[:]
    for i in index_param_to_delete:
        W_modified = np.delete(W_modified, i, 1)
        phi_modified = np.delete(phi_modified, i, 0)
        names_modified = np.delete(names_modified, i, 0)

    print('shape of W_m:\t', W_modified.shape)
    print('shape of phi_m:\t', np.array(phi_modified).shape)


    # ========== Step 6 - QR decomposition + pivoting

    (Q, R, P) = sp.qr(W_modified, pivoting=True)

    # ========== Step 7 - Calculate base parameters
    # tmp = 0

    # for i in range(np.diag(R).shape[0]):
    #         if abs(np.diag(R)[i]) < threshold:
    #             tmp = i

    R1 = R[:indexQR, :indexQR]
    R2 = R[:indexQR, indexQR:]
    Q1 = Q[:, :indexQR]
    
    for i in (indexQR, len(P)-1):
        names.pop(P[i])
    print('shape of R1',np.array(R1).shape)
    print('shape of R2',np.array(R2).shape)

    beta = np.dot(np.linalg.pinv(R1), R2)
    print('Shape of beta:\t', np.array(beta).shape)

    # ========== Step 8 - Calculate the Phi modified

    phi_base = np.dot(np.linalg.pinv(R1), np.dot(Q1.T,tau))  # Base parameters
    print('shape of phi_base:\t', np.array(phi_base).shape)

    W_base = np.dot(Q1, R1)                             # Base regressor
    print('Shape of W_base:\t', np.array(W_base).shape)
    cond=np.linalg.cond(W_base)
    print('conditionnement******************',cond)
    inertialParameters = {names_modified[i]: phi_base[i]
                        for i in range(len(phi_base))}
    


    params_rsortedphi = [] # P donne les indice des parametre par ordre decroissant 
    params_rsortedname=[]
    for ind in P:
        params_rsortedphi.append(phi_modified[ind])
        params_rsortedname.append(names_modified[ind])


    params_idp_val = params_rsortedphi[:indexQR]
    params_rgp_val = params_rsortedphi[indexQR]
    params_idp_name =params_rsortedname[:indexQR]
    params_rgp_name = params_rsortedname[indexQR]
    params_base = []
    params_basename=[]

    for i in range(indexQR):
    # for i in range(tmp+1):
        if beta[i] == 0:
            params_base.append(params_idp_val[i])
            params_basename.append(params_idp_name[i])

        else:
            params_base.append(params_idp_val[i] +  ((round(float(beta[i]), 6))*params_rgp_val))
            params_basename.append(str(params_idp_name[i]) + ' + '+str(round(float(beta[i]), 6)) + ' * ' + str(params_rgp_name))
    print('\n')
    # display of the base parameters and their identified values 
    print('base parameters and their identified values:')
    print(params_basename)
    print(params_base)
    print('\n')
    print('shape of bqse param vector',np.array(params_base).shape)
    # calculation of the torque vector using the base regressor and the base parameter 
    tau_param_base=np.dot(W_base,params_base)
    
    tau1=tau_param_base[0:tau_param_base.size -5:6]
    tau2=tau_param_base[1:tau_param_base.size -4:6]
    tau3=tau_param_base[2:tau_param_base.size -3:6]
    tau4=tau_param_base[3:tau_param_base.size -2:6]
    tau5=tau_param_base[4:tau_param_base.size -1:6]#[start:stop:step]
    tau6=tau_param_base[5:tau_param_base.size -0:6]
    # print('shape of tau1 <3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<',np.array(tau1).shape)
    # reshaping tau 
    tau_reshape=[]
    tau_reshape.extend(tau1)
    tau_reshape.extend(tau2)
    tau_reshape.extend(tau3)
    tau_reshape.extend(tau4)
    tau_reshape.extend(tau5)
    tau_reshape.extend(tau6)
    
    tau_reshape=np.array(tau_reshape)


    # print('shape of tau_reshape <3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<3<',tau_reshape.shape)
    


    return W_base,phi_base,tau_reshape

def filter_butterworth(sampling_freq,f_coupure,signale):
    sfreq = sampling_freq
    f_p = f_coupure
    nyq=sfreq/2
    
    sos = signal.iirfilter(5, f_p / nyq, btype='low', ftype='butter', output='sos')
    signal_filtrer = signal.sosfiltfilt(sos, signale)

    return signal_filtrer
    
def plot_torque_qnd_error(tau,tau_param_base):
    
    samples = []
    for i in range(tau.size):
            samples.append(i)

    # if we use W_modified=w_pin the we plot tau_pin (generated by Pin)
    # if we use W_modified=w the we plot tau(than's data file)

    plt.figure('torque pin/than et torque base parameters')
    # plt.plot(samples, tau_pin, 'g', linewidth=2, label='tau')
    plt.plot(samples, tau, 'g', linewidth=1, label='tau')
    plt.plot(samples,tau_param_base, 'b', linewidth=1, label='tau base param ')
    plt.title('tau tau_estime with base param ')
    plt.xlabel('Samples')
    plt.ylabel('torque(N/m)')
    plt.legend()
    plt.show()

    
    err = []
    for i in range(np.array(tau).size):
        err.append(abs(tau[i] - tau_param_base[i]) * abs(tau[i] - tau_param_base[i]))
    plt.plot(samples, err, linewidth=2, label="err")
    plt.title("erreur quadratique")
    plt.legend()
    plt.show()

def iden_model_v2(model, data, Q_total, V_total, A_total, param):
#This function calculates joint torques and generates the joint torque regressor.
#Note: a parameter Friction as to be set to include in dynamic model
#Input: model, data: model and data structure of robot from Pinocchio
#q, v, a: joint's position, velocity, acceleration
#N : number of samples
#nq: length of q
#Output: tau: vector of joint torque
#W : joint torque regressor"""
    nbSamples=np.array(Q_total[0]).size
    q=np.array(Q_total)
    dq=np.array(V_total)
    ddq=np.array(A_total)
    
    param['NbSample']=int(nbSamples)

    tau = np.empty(model.nq*param['NbSample'])
    W = np.empty([param['NbSample']*model.nq ,10*model.nq])
    for i in range(param['NbSample']):
        # tau_temp = pin.rnea(model, data, q[i, :], dq[i, :], ddq[i, :])
        W_temp = pin.computeJointTorqueRegressor(
            model, data, q[i, :], dq[i, :], ddq[i, :])
        for j in range(model.nq):
            # tau[j*param['NbSample'] + i] = tau_temp[j]
            W[j*param['NbSample'] + i, :] = W_temp[j, :]

    if param['Friction']:
        W = np.c_[W, np.zeros([param['NbSample']*model.nq, 2*model.nq])]
        for i in range(param['NbSample']):
            for j in range(model.nq):
                tau[j*param['NbSample'] + i] = tau[j*param['NbSample'] + i] + dq[i, j]*param['fv'] + np.sign(dq[i, j])*param['fc']
                W[j*param['NbSample'] + i, 10*model.nq+2*j] = dq[i, j]
                W[j*param['NbSample'] + i, 10*model.nq+2*j + 1] = np.sign(dq[i, j])

    return tau, W



if __name__ == "__main__":

    nbr_of_joint=6
    axe2axe_palier_de_vitesse_all_joint_one_by_one()
    Q_filtrer=[[],[],[],[],[],[]]
    V_filtrer=[[],[],[],[],[],[]]
    A_filtrer=[[],[],[],[],[],[]]

    Q_total,V_total,dq_th,ddq_th=read_tau_q_dq_ddq_fromTxt(nbr_of_joint)# gazebo
    # tau_simu_mauvais_ordre=tau_simu_mauvais_ordre
    tau_simu_mauvais_ordre=filter_butterworth(int (1/Tech),5,tau_simu_mauvais_ordre)
    # tau_filtrer=tau_simu_par_ordre
    tau_filtrer=filter_butterworth(int (1/Tech),5,tau_simu_par_ordre)
    
    for i in range(6):
        
        Q_filtrer[i]=filter_butterworth(int (1/Tech),5,Q_total[i])
        V_filtrer[i]=filter_butterworth(int (1/Tech),5,V_total[i])
        dq_th[i]=filter_butterworth(int (1/Tech),5,dq_th[i])
        ddq_th[i]=filter_butterworth(int (1/Tech),5,ddq_th[i])

    # plot_QVA_total([],nbr_of_joint,Q_filtrer,V_filtrer,ddq,'name')
    # for i in range(Q_filtrer[0].size):
    #     robot.display(np.array(Q_filtrer)[:,i])
    #     sleep(0.8)
    
    plot_QVA_total([],nbr_of_joint,Q_filtrer,dq_th,ddq_th,'name')
    
    tau, W =iden_model_v2(model, data, Q_filtrer, dq_th, ddq_th, param)

    w_pin=np.double(W)
    p_pin=np.dot(w_pin.transpose(),w_pin)
    q_pin= -np.dot(tau_filtrer.transpose(),w_pin)
    p_pin=nearestPD(p_pin)

    phi_etoile_pin=qpsolvers.solve_qp(
                p_pin,
                q_pin,
                G=None,
                h=None,
                A=None,
                b=None,
                lb=None,
                ub=None,
                solver="quadprog",
                initvals=None,
                sym_proj=True
                )

    print('*****************************************')
    # print('phi_etoile',phi_etoile.shape)
    phi_etoile_pin=np.array(phi_etoile_pin)
    phi_etoile_pin=np.double(phi_etoile_pin)
    print('phi_etoile_pin',phi_etoile_pin.shape)
    print('phi_etoile_pin_yaskawa value',phi_etoile_pin)
    print('*****************************************')

    tau_estime_pin=np.dot(w_pin,phi_etoile_pin)
    print('shape of tau_estime',tau_estime_pin.shape)

    samples = []
    for i in range(np.array(tau_filtrer).size):
        samples.append(i)

    plt.figure('torque estime et torque mesure par le robot')
    # plt.plot(samples, tau_pin, 'g', linewidth=2, label='tau')
    plt.plot(samples, tau_filtrer, 'g', linewidth=1, label='tau_robot')
    plt.plot(samples,tau_estime_pin, 'b', linewidth=1, label='tau estime ')
    plt.title('tau robot et tau estime')
    plt.xlabel(' Samples')
    plt.ylabel('parametres')
    plt.legend()
    plt.show()
    
    err = []
    for i in range(np.array(tau_filtrer).size):
        err.append(abs(tau_filtrer[i] - tau_estime_pin[i]) * abs(tau_filtrer[i] - tau_estime_pin[i]))
    plt.plot(samples, err, linewidth=2, label="err")
    plt.title("erreur quadratique")
    plt.legend()
    plt.show()









    W_base,phi_base,tau_param_base_reshaped=Base_regressor(Q_filtrer,dq_th,ddq_th,tau_simu_mauvais_ordre)
   
    plot_torque_qnd_error(tau_param_base_reshaped,tau_filtrer)

    genrate_W_And_torque_experimentale(Q_filtrer,dq_th,ddq_th,tau_simu_mauvais_ordre)
    


'''
# initialisation
q_start=[]
q_end=[]
Vmax=[]
acc_max=[]
Tech=0.0001

# nomber of joint initialisation
nbrjoint=2

#Data calculation for all joint 
time_tau,time_final,q_start,q_end,Vmax,acc_max,D=Data_Alltrajectory(nbrjoint,Tech)

# calulation and display of theoratical velocity
plt.figure('V velocity theoratical')
for i in range(nbrjoint):
    v1,time1=PreDefined_velocity_trajectory(time_tau,time_final,Vmax[i],acc_max[i],Tech)
    plt.plot(time1,v1,linewidth=1, label='V'+str(i))
plt.title('V velocity th')
plt.xlabel('t')
plt.ylabel('V')
plt.legend()
plt.show()


# calculation and display for all trajectorys
plt.figure('q Trajectory')
for i in range(nbrjoint):
    q1,time1,v1,a1=PreDefined_trajectory(time_tau,time_final,q_start[i],q_end[i], Vmax[i],acc_max[i],D[i],Tech)
    plt.plot(time1,q1,linewidth=1, label='q'+str(i))

plt.title('q Trajectory')
plt.xlabel('t')
plt.ylabel('q')
plt.legend()
plt.show()



#calculation and Display of all velocity (dq)

plt.figure('V velocity calculated via derivation(dq)')
for i in range(nbrjoint):
    q1,time1,v1,a1=PreDefined_trajectory(time_tau,time_final,q_start[i],q_end[i], Vmax[i],acc_max[i],D[i],Tech)
    plt.plot(time1,v1,linewidth=1, label='V'+str(i))
plt.xlabel('t')
plt.ylabel('V')
plt.legend()
plt.show()
plt.title('V velocity calculated via derivation(dq)')



#calculation of trajectory with several value of velocity
# q_max,time,v_max,a_max=trajectory(10,150,12,2,Tech)
# q1_20,time1,v1_20,a1_20=trajectory(0,100,14,3,Tech)
# q2_40,time2,v2_40,a2_40=trajectory(q_start,q_end,Vmax*0.4,acc_max,Tech)
# q3_60,time3,v3_60,a3_60=trajectory(q_start,q_end,Vmax*0.6,acc_max,Tech)
# q4_80,time4,v4_80,a4_80=trajectory(q_start,q_end,Vmax*0.8,acc_max,Tech)

#Display of trajectory

# plt.figure('q Trajectory')
# plt.plot(time,q_max,  linewidth=1, label='10-150-12-2 position')
# plt.plot(time1,q1_20, linewidth=1, label='0-100-14-2 position')
# # plt.plot(time2,q2_40, linewidth=1, label='q2_40 position')
# plt.plot(time3,q3_60, linewidth=1, label='q3_60 position')
# plt.plot(time4,q4_80, linewidth=1, label='q4_80 position')
# plt.title('q Trajectory')
# plt.xlabel('t')
# plt.ylabel('q')
# plt.legend()
# plt.show()

# #Display of velocity 

# plt.figure('V velocity')
# plt.plot(time,v_max,linewidth=1, label='v_max ')
# plt.plot(time1,v1_20, linewidth=1, label='v1_20 ')
# # plt.plot(time2,v2_40, linewidth=1, label='v2_40 ')
# # plt.plot(time3,v3_60, linewidth=1, label='v3_60 ')
# # plt.plot(time4,v4_80, linewidth=1, label='v4_80 ')
# plt.title('V velocity')
# plt.xlabel('t')
# plt.ylabel('V')
# plt.legend()
# plt.show()

# #Display of acceleration 

# plt.figure('a acceleration')
# plt.plot(time,a_max,linewidth=1, label='a_max ')
# plt.plot(time1,a1_20, linewidth=1, label='a1_20 ')
# plt.plot(time2,a2_40, linewidth=1, label='a2_40 ')
# plt.plot(time3,a3_60, linewidth=1, label='a3_60 ')
# plt.plot(time4,a4_80, linewidth=1, label='a4_80 ')
# plt.title('a acceleration')
# plt.xlabel('t')
# plt.ylabel('a')
# plt.legend()
# plt.show()

'''
