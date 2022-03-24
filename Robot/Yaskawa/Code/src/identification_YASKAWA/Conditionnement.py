from ctypes import sizeof
from operator import index
from pyexpat import model
from termios import TCSAFLUSH
from numpy import double, linalg, math, sign, sqrt, transpose
from numpy.core.fromnumeric import shape
from numpy.lib.nanfunctions import _nanmedian_small
#from Robot.Yaskawa.Code.src.identification_YASKAWA.Trajectoire_yaskawa_v2 import Q_total
from pinocchio.visualize import GepettoVisualizer
from pinocchio.robot_wrapper import RobotWrapper
import matplotlib.pyplot as plt
import scipy.linalg as sp
import pinocchio as pin
import numpy as np
from tabulate import tabulate
import os
from typing import Any, Optional
from typing import Optional
import qpsolvers
from time import sleep
import random

#----------------------------------------------------
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
#------------------------------------------------------
def standardParameters(model, param):
#This function prints out the standard inertial parameters obtained from 3D design.
#Note: a flag IsFrictioncld to include in standard parameters
#Input: njoints: number of jointsOutput: params_std: a dictionary of parameter names and their values
    params_name = ['m', 'mx', 'my', 'mz', 'Ixx','Ixy', 'Iyy', 'Ixz', 'Iyz', 'Izz']
    phi = []
    params = []
    for i in range(1, model.njoints):
        P = model.inertias[i].toDynamicParameters()
        for k in P:
            phi.append(k)
        for j in params_name:
            params.append(j + str(i))

    if param['Friction']:
        for k in range(1, model.njoints):
            # Here we add arbitrary values

            phi.extend([param['fv'], param['fc']])
            params.extend(['fv' + str(k), 'fc' + str(k)])
    params_std = dict(zip(params, phi))
    return params_std

def iden_model(model, data, q, dq, ddq, param):
#This function calculates joint torques and generates the joint torque regressor.
#Note: a parameter Friction as to be set to include in dynamic model
#Input: model, data: model and data structure of robot from Pinocchio
#q, v, a: joint's position, velocity, acceleration
#N : number of samples
#nq: length of q
#Output: tau: vector of joint torque
#W : joint torque regressor"""

    tau = np.empty(model.nq*param['NbSample'])
    W = np.empty([param['NbSample']*model.nq ,10*model.nq])
    for i in range(param['NbSample']):
        tau_temp = pin.rnea(model, data, q[i, :], dq[i, :], ddq[i, :])
        W_temp = pin.computeJointTorqueRegressor(
            model, data, q[i, :], dq[i, :], ddq[i, :])
        for j in range(model.nq):
            tau[j*param['NbSample'] + i] = tau_temp[j]
            W[j*param['NbSample'] + i, :] = W_temp[j, :]

    if param['Friction']:
        W = np.c_[W, np.zeros([param['NbSample']*model.nq, 2*model.nq])]
        for i in range(param['NbSample']):
            for j in range(model.nq):
                tau[j*param['NbSample'] + i] = tau[j*param['NbSample'] + i] + dq[i, j]*param['fv'] + np.sign(dq[i, j])*param['fc']
                W[j*param['NbSample'] + i, 10*model.nq+2*j] = dq[i, j]
                W[j*param['NbSample'] + i, 10*model.nq+2*j + 1] = np.sign(dq[i, j])

    return tau, W

def eliminateNonAffecting(W_, params_std, tol_e):
#This function eliminates columns which has L2 norm smaller than tolerance.
#Input: W: joint torque regressor
#tol_e: tolerance
#Output: W_e: reduced regressor
#params_r: corresponding parameters to columns of reduced regressor"""
    col_norm = np.diag(np.dot(W_.T, W_))
    idx_e = []
    params_e = []
    params_r = []
    for i in range(col_norm.shape[0]):
        if col_norm[i] < tol_e:
            idx_e.append(i)
            params_e.append(list(params_std.keys())[i])
        else:
            params_r.append(list(params_std.keys())[i])
    
    W_e = np.delete(W_, idx_e, 1)
    return W_e, params_r,idx_e

def double_QR(tau, W_e, params_r, params_std=None):
#This function calculates QR decompostion 2 times, first to find symbolic
#expressions of base parameters, second to find their values after re-organizing
#regressor matrix.
#Input: W_e: regressor matrix (normally after eliminating zero columns)
#params_r: a list of parameters corresponding to W_e
#Output: W_b: base regressor
#base_parametes: a dictionary of base parameters"""
# scipy has QR pivoting using Householder reflection
    Q, R = np.linalg.qr(W_e)

    # sort params as decreasing order of diagonal of R
    assert np.diag(R).shape[0] == len(
        params_r
    ), "params_r does not have same length with R"

    idx_base = []
    idx_regroup = []

    # find rank of regressor
    epsilon = np.finfo(float).eps  # machine epsilon
    tolpal = W_e.shape[0]*abs(np.diag(R).max()) * \
        epsilon  # rank revealing tolerance
    # tolpal = 0.02
    for i in range(len(params_r)):
        if abs(np.diag(R)[i]) > tolpal:
            idx_base.append(i)
        else:
            idx_regroup.append(i)

    numrank_W = len(idx_base)

    # rebuild W and params after sorted
    W1 = np.zeros([W_e.shape[0], len(idx_base)])
    W2 = np.zeros([W_e.shape[0], len(idx_regroup)])

    params_base = []
    params_regroup = []
    print(idx_base,"ICI")
    for i in range(len(idx_base)):
        W1[:, i] = W_e[:, idx_base[i]]
        params_base.append(params_r[idx_base[i]])
    for j in range(len(idx_regroup)):
        W2[:, j] = W_e[:, idx_regroup[j]]
        params_regroup.append(params_r[idx_regroup[j]])

    W_regrouped = np.c_[W1, W2]

    # perform QR decomposition second time on regrouped regressor
    Q_r, R_r = np.linalg.qr(W_regrouped)

    R1 = R_r[0:numrank_W, 0:numrank_W]
    Q1 = Q_r[:, 0:numrank_W]
    R2 = R_r[0:numrank_W, numrank_W: R.shape[1]]

    print(Q1.shape)
    print(np.linalg.matrix_rank(Q1))

    # regrouping coefficient
    beta = np.around(np.dot(np.linalg.inv(R1), R2), 6)

    # values of base params
    phi_b = np.round(np.dot(np.linalg.inv(R1), np.dot(Q1.T, tau)), 6)
    print(phi_b)

    # base regressor
    W_b = np.dot(Q1, R1)

    '''phi_pinv=np.round(np.matmul(np.linalg.pinv(W_b),tau), 6)
    print(phi_pinv)
    phi_b=phi_pinv[0:len(phi_b)]'''

    assert np.allclose(W1, W_b), "base regressors is wrongly calculated!  "

    # reference values from std params
    if params_std is not None:
        phi_std = []
        for x in params_base:
            phi_std.append(params_std[x])
        for i in range(numrank_W):
            for j in range(beta.shape[1]):
                phi_std[i] = phi_std[i] + beta[i, j] * \
                    params_std[params_regroup[j]]
        phi_std = np.around(phi_std, 5)

    tol_beta = 1e-6  # for scipy.signal.decimate
    for i in range(numrank_W):
        for j in range(beta.shape[1]):
            if abs(beta[i, j]) < tol_beta:

                params_base[i] = params_base[i]

            elif beta[i, j] < -tol_beta:

                params_base[i] = (
                    params_base[i]
                    + " - "
                    + str(abs(beta[i, j]))
                    + "*"
                    + str(params_regroup[j])
                )

            else:

                params_base[i] = (
                    params_base[i]
                    + " + "
                    + str(abs(beta[i, j]))
                    + "*"
                    + str(params_regroup[j])
                )
    base_parameters = dict(zip(params_base, phi_b))
    print('base parameters and their identified values: ')
    table = [params_base, phi_b]
    print(tabulate(table))

    if params_std is not None:
        return W_b, base_parameters, params_base, phi_b, phi_std
    else:
        return W_b, base_parameters, params_base, phi_b,idx_base

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

def Generate_posture_static_Fadi():
    
    # Q_total=[[],[],[],[],[],[]]
    Q_total=[]
    Q_total=np.array(Q_total)
    posture1=np.array([[0],[0],[0],[0],[0],[0]])
    Q_total=posture1
    #print("shape of posture 1",np.array(posture1).shape)

    posture1=np.array([[0],[0],[4.71238898038],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture1], axis=1)

    posture2=np.array([[3.1415],[0],[3.1415],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture2], axis=1)

    posture3=np.array([[0],[0],[1.57079632679],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture3], axis=1)

    posture4=np.array([[2.356125 ],[0.25],[3.1415],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture4], axis=1)

    posture5=np.array([[0],[0],[0.78539816339],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture5], axis=1)

    posture6=np.array([[2.356125],[0.76],[3.1415],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture6], axis=1)

    posture7=np.array([[0],[0],[2.09439510239],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture7], axis=1)

    posture8=np.array([[3.1415],[0],[4.71225],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture8], axis=1)

    posture9=np.array([[0],[0.78539816339],[4.71238898038],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture9], axis=1)

    posture10=np.array([[3.1415],[0],[1.57075],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture10], axis=1)

    posture11=np.array([[3.1415],[0],[1.9],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture11], axis=1)

    posture12=np.array([[3.1415],[0],[3.1415],[3.1415],[0],[0]])
    Q_total=np.concatenate([Q_total,posture12], axis=1)

    posture13=np.array([[3.1415],[0],[3.1415],[0],[-3.1415],[0]])
    Q_total=np.concatenate([Q_total,posture13], axis=1)

    posture14=np.array([[0],[1.0471975512],[4.71238898038],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture14], axis=1)

    posture15=np.array([[0],[0.52359877559],[3.1415],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture15], axis=1)

    posture16=np.array([[0],[0.52359877559],[4.71238898038],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture16], axis=1)

    posture17=np.array([[0],[0.52359877559],[1.57079632679],[0.78539816339],[3.0],[0]])
    Q_total=np.concatenate([Q_total,posture17], axis=1)

    posture18=np.array([[1.57075],[-1.57075],[3.141592653589793],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture18], axis=1)

    posture19=np.array([[1.57075],[-1.57075],[1.57075],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture19], axis=1)

    posture20=np.array([[1.57075],[1.57075],[4.71238898038],[0],[0],[0]])
    Q_total=np.concatenate([Q_total,posture20], axis=1)


    return Q_total
#===============MAIN==========================================================
#=============================================================================
if __name__=="__main__": 
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

#====GENERATE RANDOM Q AND STATIC POS==========================
    q_random= np.random.rand(param['NbSample'], NQ) * np.pi - np.pi/2
    dq = np.zeros((param['NbSample'], NQ))# matrice des zero 6 lignes nbr de posture en colones 
    ddq = np.zeros((param['NbSample'], NQ)) # matrice des zero 6 lignes nbr de posture en colones

#====FIRST REALISATION==========================
    Tau, W = iden_model(model, data, q_random, dq, ddq, param)
    params_std = standardParameters(model, param)
    W_e, params_r,idx_elim1 = eliminateNonAffecting(W,params_std,tol_e=0.0001)
    W_b, base_parameters, params_base, phi_b,idx_base = double_QR(Tau, W_e, params_r, params_std=None)
    Cond = linalg.cond(W_b)
    print("Conditionnement",Cond)
#===TEST WITH POSTURE============================
    Q_pos=[]
    Q_pos = Generate_posture_static_Fadi()
    random.shuffle(Q_pos)

#=====GEPETTO==================================
    for i in range(Q_pos[0].size):
        robot.display(Q_pos[:,i])
        sleep(0.1)
    Q_pos = np.transpose(Q_pos)

    paramPosture={
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
    paramPosture['NbSample']= Q_pos.shape[0]
#=======W_BASE FOR POSTURE
    Tau_Posture, W_Posture = iden_model(model, data, Q_pos, dq, ddq, paramPosture)
    Wfirstelim_Posture = np.delete(W_Posture, idx_elim1, 1)
    W_Posture_base = np.delete(Wfirstelim_Posture, [2,5,8,9,11,13,16], 1) #[2,5,8,9,11,13,16] invers idx_base
    
#===CALCUL COND FOR POSTURE
    Cond_posture = []
    Exps = []
    M_for_Cond_Base = np.array(W_Posture_base[0:18])
    Exps.append(3)
    Cond_posture.append(linalg.cond(M_for_Cond_Base))

    print(W_Posture_base.shape[0])
    for i in range(18,W_Posture_base.shape[0]-5,6):  
        #print("IM IN",i)
        AddExp = np.array(W_Posture_base[i:i+6])
        M_for_Cond_Base = np.concatenate((M_for_Cond_Base,AddExp))
        Exps.append(i/6+1)
        Cond_posture.append(linalg.cond(M_for_Cond_Base))
    

    #print("First Cond",linalg.cond(W_Posture_base[0:18]))
    #print("Last Cond",linalg.cond(W_Posture_base))
    print(Cond_posture)
    #print(Exps)
    
    
    plt.figure('Conditionnement en fonction du nombre dexps')
    plt.plot(Exps, Cond_posture, 'g', linewidth=2, label='Conditionnement')
    plt.title('Conditionnement en fonction du nombre dexps')
    plt.xlabel('NbrsExps')
    plt.ylabel('Cond')
    plt.legend()
    plt.yscale('log')
    plt.show()

