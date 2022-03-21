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
from tabulate import tabulate
import os
from typing import Optional
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

    numrank_W = len(idx_base)

    # rebuild W and params after sorted
    W1 = np.zeros([W_e.shape[0], len(idx_base)])
    W2 = np.zeros([W_e.shape[0], len(idx_regroup)])

    params_base = []
    params_regroup = []
    print(idx_base)
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
        return W_b, base_parameters, params_base, phi_b

#=============================================================================
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

    nbSamples = param['NbSample']
    q_random= np.random.rand(nbSamples, NQ) * np.pi - np.pi/2
    dq = np.zeros((nbSamples, NQ))# matrice des zero 6 lignes nbr de posture en colones 
    ddq = np.zeros((nbSamples, NQ)) # matrice des zero 6 lignes nbr de posture en colones

    Tau, W = iden_model(model, data, q_random, dq, ddq, param)
    print('shape of Tau',np.array(Tau).shape)
    print('shape of W',np.array(W).shape)

    
