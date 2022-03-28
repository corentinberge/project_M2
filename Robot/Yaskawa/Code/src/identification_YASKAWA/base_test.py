from cmath import tau
from ctypes import sizeof
from operator import index
from pyexpat import model
from random import random
from termios import TCSAFLUSH
from turtle import Shape
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
# from tabulate import tabulate
import os
from typing import Any, Optional
from typing import Optional
import qpsolvers
from time import sleep
import random
from Trajectoire_yaskawa_v2 import read_tau_q_dq_ddq_fromTxt
from Trajectoire_yaskawa_v2  import filter_butterworth
from Trajectoire_yaskawa_v2 import plot_torque_qnd_error
from Trajectoire_yaskawa_v2 import plot_QVA_total
from Trajectoire_yaskawa_v2 import estimation_with_qp_solver
from Conditionnement import Generate_posture_static

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
# 
# 
# idx_e_predifine=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 19, 22, 24, 25, 26, 27, 28, 29, 34, 35,
#        36, 37, 38, 39, 44, 45, 46, 47, 48, 49, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
# idx_base_predifine=[0, 1, 3, 4, 6, 7, 10, 12, 14, 15]
# numrank_W_predifine =10
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
        'SAVE_FILE':0,
        'isFrictionincld':1
    }
param['NbSample']=int (param['tf']/param['ts'])

def standardParameters(model, param):
    """This function prints out the standard inertial parameters obtained from 3D design.
            Note: a flag IsFrictioncld to include in standard parameters
            Input: 	njoints: number of joints
            Output: params_std: a dictionary of parameter names and their values"""
    params_name = ['m', 'mx', 'my', 'mz', 'Ixx',
                   'Ixy', 'Iyy', 'Ixz', 'Iyz', 'Izz']
    phi = []
    params = []

    for i in range(1,model.njoints):
        P = model.inertias[i].toDynamicParameters()
        for k in P:
            phi.append(k)
        for j in params_name:
            params.append(j + str(i))

    if param['isFrictionincld']:
        for k in range(1, model.njoints):
            # Here we add arbitrary values
            
            phi.extend([param['fv'], param['fc']])
            params.extend(['fv' + str(k), 'fc' + str(k)])
    params_std = dict(zip(params, phi))
    print('shape of phi',np.array(phi).shape)
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
    print(col_norm.shape[0])
    for i in range(col_norm.shape[0]):
        print(i,len(list(params_std.keys())))
        if col_norm[i] < tol_e:
            idx_e.append(i)
            params_e.append(list(params_std.keys())[i])
        else:
            params_r.append(list(params_std.keys())[i])
    # idx_e=idx_base_predifine
    # print('idx_e',idx_e)
    W_e = np.delete(W_, idx_e, 1)
    return W_e, params_r

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

    # idx_base=idx_base_predifine

    numrank_W = len(idx_base)
    # numrank_W=numrank_W_predifine
    # rebuild W and params after sorted
    W1 = np.zeros([W_e.shape[0], len(idx_base)])
    W2 = np.zeros([W_e.shape[0], len(idx_regroup)])

    params_base = []
    params_regroup = []
    print('inx de base',idx_base)
    print('numrank_W',numrank_W)
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
    #print(tabulate(table))
    print(table)
    if params_std is not None:
        return W_b, base_parameters, params_base, phi_b, phi_std
    else:
        return W_b, base_parameters, params_base, phi_b

#=============================================================================
#=============================================================================
if __name__=="__main__": 

    # nbSamples = param['NbSample']
    # q= np.random.rand(nbSamples, NQ) * np.pi - np.pi/2
    # dq= np.zeros((nbSamples, NQ))# matrice des zero 6 lignes nbr de posture en colones 
    # ddq= np.zeros((nbSamples, NQ))# matrice des zero 6 lignes nbr de posture en colones

    param['NbSample']=int(1264)#size of q (number of sampels)
    nbSamples = param['NbSample']

    q,dq,ddq,tau_robot=read_tau_q_dq_ddq_fromTxt(6)
    # dq= np.zeros((nbSamples, NQ))# matrice des zero 6 lignes nbr de posture en colones 
    # ddq= np.zeros((nbSamples, NQ))# matrice des zero 6 lignes nbr de posture en colones
    print('shape of q',np.array(q).shape)
    print('shape of tau_robot',tau_robot.shape)
    
    plot_QVA_total([],6,(q.T),(dq.T),(ddq.T),'joint')
    
    param_std= standardParameters(model, param)
    
    _, W=iden_model(model, data, q, dq, ddq, param)
    
    W_e, params_r=eliminateNonAffecting(W, param_std, 0.00001)
    
    W_b, base_parameters, params_base, phi_b, phi_std=double_QR(tau_robot, W_e, params_r, param_std)
    con=np.linalg.cond(W_b)

    print('lez conditionnement est ',con)

    tau_base=np.dot(W_b,phi_b)
    
    # calcul pour les parametres de base et regresseur de base
    plot_torque_qnd_error(tau_robot,tau_base)

    #calcul pour tous les parametre et le regresseur initiale
    estimation_with_qp_solver(W,tau_robot)
    
    
    
