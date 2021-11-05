from numpy import linalg
from numpy.core.fromnumeric import shape
from numpy.lib.nanfunctions import _nanmedian_small
from pinocchio.visualize import GepettoVisualizer
from pinocchio.robot_wrapper import RobotWrapper
import matplotlib.pyplot as plt
import scipy.linalg as sp
import pinocchio as pin
import numpy as np
import os


def initialize():
    """
    Import and initialize the robot

    Returns
    -------
    robot: RoborWrapper
        Robot generated with urdf file
    """
    package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = package_path + '/robots/urdf/planar_2DOF.urdf'

    robot = RobotWrapper()
    robot.initFromURDF(urdf_path, package_path, verbose=True)
    # robot.initViewer(loadModel=True)
    # robot.display(robot.q0)

    return robot

# =====================
# |  Base functions   |
# =====================

def generateParameters(model, NJOINT):
    """
    Generate inertial parameters for all links (excepted the base link) and their associated names

    Parameters
    ----------
    model: RobotWrapper.model
        robot model
    NJOINT: int
        number of robot joints

    Returns
    -------
    (names, phi): arrays tuples
        names: string array
            names of parameters
        phi: float array
            values of parameters
    """
    names = []
    for i in range(1, NJOINT):
        names += ['m'+str(i), 'mx'+str(i), 'my'+str(i), 'mz'+str(i), 'Ixx'+str(i),
                  'Ixy'+str(i), 'Iyy'+str(i), 'Izx'+str(i), 'Izy'+str(i), 'Izz'+str(i)]

    phi = []
    for i in range(1, NJOINT):
        phi.extend(model.inertias[i].toDynamicParameters())

    return np.array(names), np.array(phi)


def generateRandomInputsAndTorque(model, data, NQ, nbSamples):
    """
    Generate a certain amount of random position, speed and aceleration and calculate torque for each joint and sample

    Parameters
    ----------
    model: RobotWrapper.model
        robot model
    data: RobotWrapper.data
        robot data
    NQ: int
        number of robot joints
    nbSamples: int
        number of samples to generate

    Returns
    -------
    (imputs, tau): tuple
        imputs: tuple of arrays
            (position, speed, acceleration)
        tau: float array
            torque array (nbSamples*NQ)  
    """
    q = np.random.rand(NQ, nbSamples) * np.pi - np.pi/2  # -pi/2 < q < pi/2
    dq = np.random.rand(NQ, nbSamples) * 10              # 0 < dq  < 10
    ddq = np.ones((NQ, nbSamples))                       # ddq = 1

    tau = []
    for i in range(nbSamples):
        tau.extend(pin.rnea(model, data, q[:, i], dq[:, i], ddq[:, i]))

    return (q, dq, ddq), np.array(tau)


def generateRegressor(model, data, inputs, nbSamples):
    """
    Generate regressor based on inputs and the number of samples

    Parameters
    ----------
    model: RobotWrapper.model
        robot model
    data: RobotWrapper.data
        robot data
    inputs: float array tuple
        tuple of (position, speed, acceleration)
    nbSamples: int
        number of samples to generate

    Returns
    -------
    W: float array
        regressor
    """
    q, dq, ddq = inputs
    W = []  # Regression vector
    for i in range(nbSamples):
        W.extend(pin.computeJointTorqueRegressor(
            model, data, q[:, i], dq[:, i], ddq[:, i]))
    return np.array(W)


def removeZeroParameters(W, phi, names, threshold):
    """
    Remove zero parameters and update phi and names of parameters

    Parameters
    ----------
    W: float array
        regressor
    phi: float array
        values of parameters
    names: string arrays
        names of parameters
    threshold: float or double
        threshold below which parameters are considered as zero

    Returns
    -------
    (W, phi, names): tuple of arrays
        W: float array
            regressor
        phi: float array
            parameters values
        names: string array
            parameters names

    """
    W_modified = np.array(W[:])
    tmp = []
    for i in range(len(phi)):
        if (np.dot([W_modified[:, i]], np.transpose([W_modified[:, i]]))[0][0] <= threshold):
            tmp.append(i)
    tmp.sort(reverse=True)

    phi_modified = phi[:]
    names_modified = names[:]
    for i in tmp:
        W_modified = np.delete(W_modified, i, 1)
        phi_modified = np.delete(phi_modified, i, 0)
        names_modified = np.delete(names_modified, i, 0)

    return W_modified, phi_modified, names_modified


def calculateBaseParam(Q, R, P, names, threshold):
    """
    Calculate base parameters from the pivoted QR (and P) decomposition of the regressor

    Parameters
    ----------
    Q: float array
        Lower tiangular matrix
    R: float array
        Upper triangular matrix
    P: int array
        order of indexes after pivoting
    names: string array
        names of parameters
    threshold: float or double
        threshold below which parameters are considered as zero

    Returns
    -------
    (R1, R2, Q1): tuple of 3 float arrays
        R1 and Q1: float arrays 
            matrix with only indemendant parameters.
        R2: float array
            dependant parameters  
    """
    tmp = 0
    for i in range(len(R[0])):
        if R[i, i] > threshold:
            tmp = i

    R1 = R[:tmp+1, :tmp+1]
    R2 = R[:tmp+1, tmp+1:]

    Q1 = Q[:, :tmp+1]

    for i in (tmp+1, len(P)-1):
        names = np.delete(names, P[i])
        # names.pop(P[i])

    return R1, R2, Q1


def showTorquePlot(tau, tau_base, NQ, nbSamples):
    """
    Show torques and base torques plots

    Parameters
    ----------
    tau: float array
        initial torques
    tau_base: float array
        base torques
    NQ: int
        number of robot joints
    nbSamples: int
        number of samples
    """
    samples = []
    for i in range(nbSamples * NQ):
        samples.append(i)

    # trace le resultat dans un graph
    # les deux plot sur la memes figure
    plt.plot(samples, tau_base, color='green', linewidth=2, label="tau_base")  # linewidth linestyle
    plt.plot(samples, tau, color='blue', linewidth=1, label="tau")
    plt.legend()
    plt.title("graphe montrant tau et tau_base")

    # les deux plot sur deux figures differentes
    fig, axs = plt.subplots(2)
    fig.suptitle('tau et tau_base separement')
    axs[0].plot(samples, tau, color='blue', label="tau")
    # plt.legend()
    axs[1].plot(samples, tau_base, color='green', label="tau_base")
    plt.legend()
    # showing results
    plt.show()


def calculateEstimatedParam(W_base, tau):
    """"
    Calculate estimated parameters

    Parameters
    ----------
    W_base: float array
        base regressor
    tau: float array
        initial torques
    """
    AtA = np.linalg.inv(np.dot(W_base.T, W_base))
    AtB = np.dot(W_base.T, tau)
    return np.dot(AtA, AtB)


def calculateError(tau, tau_base, NQ, nbSamples):
    """
    Calculate error between tau and tau_base based on the identification

    Parameters
    ----------
    tau: float array
        initial torques
    tau_base: float array
        base torques
    NQ: int
        number of robot joints
    nbSamples: int
        number of samples

    Returns
    -------
    err: float array
        Quadratic error
    """
    err = []
    for i in range(nbSamples * NQ):
        err.append(abs(tau[i] - tau_base[i]) * abs(tau[i] - tau_base)[i])
    return np.array(err)

# =====================
# | Display functions |
# =====================

def showParametersAndEquationsPlot(R, P, phi, names, threshold, beta):
    """
    Show base parameters names and values and the equations

    Parameters
    ----------
    R: float array
        Upper triangular matrix
    P: int array
        order of indexes after pivoting
    phi: float array
        values of parameters
    names: string array
        names of parameters
    threshold: float or double
        threshold below which parameters are considered as zero
    beta: ???
        beta coefficient

    """
    tmp = 0
    for i in range(len(R[0])):
        if R[i, i] > threshold:
            tmp = i

    params_rsortedphi = [] # P donne les indice des parametre par ordre decroissant 
    params_rsortedname = []
    for ind in P:
        params_rsortedphi.append(phi[ind])
        params_rsortedname.append(names[ind])

    params_idp_val = params_rsortedphi[:tmp+1]
    params_rgp_val = params_rsortedphi[tmp+1]
    params_idp_name = params_rsortedname[:tmp+1]
    params_rgp_name = params_rsortedname[tmp+1]
    params_base = []
    params_basename=[]

    for i in range(tmp+1):
        if beta[i] == 0:
            params_base.append(params_idp_val[i])
            params_basename.append(params_idp_name[i])

        else:
            params_base.append(str(params_idp_val[i]) + ' + '+str(round(float(beta[i]), 6)) + ' * ' + str(params_rgp_val))
            params_basename.append(str(params_idp_name[i]) + ' + '+str(round(float(beta[i]), 6)) + ' * ' + str(params_rgp_name))

    print('base parameters and their identified values: \n')
    print(params_base)
    print('\n')
    table = [phi,params_base]
    print(table)
    print('\n')
    table1 = [names,params_basename]
    print('base_parametre and equation \n')
    print(table1)
    # print('valeurs base et calcul\t',table[i][i])
    # print('finale table shape \t', np.array(table).shape)
    # print(table)


def showPhiPlots(estimatedPhi, W_base):
    """"
    TODO: commenter
    """
    samples1 = []
    for i in range(estimatedPhi.shape[0]):
        samples1.append(i + 1)

    plt.scatter(samples1, W_base, color='green', linewidth=2, label="phi(base)")
    plt.scatter(samples1, estimatedPhi, color='red', linewidth=1, label="phi etoile")
    plt.title("graphe montrant phi et phi etoile")
    plt.legend()
    plt.show()


def showErrorPlot(err, NQ, nbSamples):
    """
    Show error plot

    Parameters
    ----------
    err: float array
        Quadratic error
    NQ: int
        number of robot joints
    nbSamples: int
        number of samples
    """
    samples = []
    for i in range(nbSamples * NQ):
        samples.append(i)

    # print(np.array(err).shape)
    plt.plot(samples, err, label="err")
    plt.title("erreur quadratique")
    plt.legend()
    plt.show()

# =====================
# |       Main        |
# =====================

def main():
    """
    Main function
    """

    # ========== Step 1 - Load model, create robot model and create robot data
    robot = initialize()
    data = robot.data
    model = robot.model
    NQ = robot.nq                 # joints angle
    NV = robot.nv                 # joints velocity
    NJOINT = robot.model.njoints  # number of links
    # gv = robot.viewer.gui # uncomment tu run this in gepetto-gui

    # ========== Step 2 - Generate inertial parameters for all links (excepted the base link)
    names, phi = generateParameters(model, NJOINT)

    # ========== Step 3 - Generate input and output - 100 samples
    nbSamples = 100  # number of samples
    inputs, tau = generateRandomInputsAndTorque(model, data, NQ, nbSamples)

    # ========== Step 4 - Create IDM with pinocchio
    regressor = generateRegressor(model, data, inputs, nbSamples)

    # ========== Step 5 - Remove non dynamic effect columns then remove zero value columns then remove the parameters
    #                     related to zero value columns at the end we will have a matix W_modified et Phi_modified
    threshold = 0.000001
    regressor, phi, names = removeZeroParameters(regressor, phi, names, threshold)

    # ========== Step 6 - QR decomposition + pivoting
    (Q, R, P) = sp.qr(regressor, pivoting=True)

    # ========== Step 7 - Calculate base parameters and beta coefficient
    R1, R2, Q1 = calculateBaseParam(Q, R, P, names, threshold)
    beta = np.dot(np.linalg.inv(R1), R2)

    # ========== Step 8 - Calculate the modified phi
    baseParameters = np.dot(np.linalg.inv(R1), np.dot(Q1.T, tau))  # Base parameters
    baseRegressor = np.dot(Q1, R1)                                 # Base regressor

    inertialParameters = {names[i]: baseParameters[i] for i in range(len(baseParameters))}
    print("Base parameters:\n", inertialParameters)
    showParametersAndEquationsPlot(R, P, phi, names, threshold, beta)

    # ========== Step 9 - Calculate tau with phi_base and base regressor
    tau_base = np.dot(baseRegressor, baseParameters)
    showTorquePlot(tau, tau_base, NQ, nbSamples)

    # ========== Step 10 - Calculate phi* least square min abs(tau - phi* * W_base)^2.
    #                      We apply classic least square method by nullifying error gradient with Error Hes > 0
    estimatedParameters = calculateEstimatedParam()
    print('shape of phi_etoile \t', estimatedParameters.shape)
    showPhiPlots()

    # ========== Step 11 - Calculate error between tau and tau_base based on the identification
    # calcul l'err entre tau et tau_base calcule a partir de l'identification
    err = calculateError()
    showErrorPlot()


if __name__ == '__main__':
    main()
