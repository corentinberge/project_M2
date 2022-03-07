from numpy.core.defchararray import join
from numpy.core.fromnumeric import transpose
import pinocchio as pin
from pinocchio.utils import *
from pinocchio.visualize import GepettoVisualizer
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import norm, inv, pinv
import time
import os

MAIN = True #Execution du main
PLOT = True #pour avoir les PLOTS


def situationOT(M):
    """ cette fonction permets à partir d'un objet SE3, d'obtenir un vecteur X contenant la transaltion et la rotation de L'OT (situation de l'OT)
    avec les angles d'euler classique, M est l'objet SE3, out = [ex ey ez psi theta phi] """
    p = M.translation
    delta = orientationEuler(M.rotation)
    return np.concatenate((p,delta),axis=0)
        

def orientationEuler(R):
    """ Renvois l'orientation selon la valeurs des angles d'euler  
    prend une matrice de rotation 3x3 en entrée"""
    if(R[2,2] != 1):
        psi = math.atan2(R[0,2],-R[1,2])
        theta = math.acos(R[2,2])
        phi = math.atan2(R[2,0],R[2,1])
    else :
        print("attention psi et phi ne sont pas définis ici ils seront pris égaux")
        a = math.atan2(R[0,1],R[0,0])/(1-R[2,2])
        psi = a
        theta = math.pi*(1-R[2,2])/2
        phi = a
    return np.array([psi,theta,phi])


def adaptSituation(X):
    """ cette fonction permets d'adapter la situation de l'organe terminal au plan, ressort un vecteur 3,3
    on enleve ey car il n'y a pas de translation selon ey, on enleve psi et phi car rotation constantes """
    nX = np.array([X[0],X[2],X[4]]) # on enleve les colonnes 2,3 et 5 donc on enlève les lignes 2,3 et 5 de la jacobienne 
    return nX


def adaptJacob(J):
    """ cette fonction permets d'adapter la jacobienne, on enlève les lignes 1,3 et 5 """
    nJ = np.array([J[0,:],J[2,:],J[4,:]])
    return nJ

def simulateurTraj(N,robot,IDX):
    dt = 1e-2
    a0,a1,a2 = 0,1,2
    X = np.zeros((N,3))
    for i in range(N-1):
        q,dq = loiPoly(a0,a1,a2,i*dt)
        q = np.array(q)
        robot.forwardKinematics(q) #update joint 
        pin.updateFramePlacements(robot.model,robot.data) #update frame placement  
        X[i,:] =  adaptSituation(situationOT(robot.data.oMf[IDX]))
        robot.display(q)
        time.sleep(dt)
    print(X.shape)
        
#Loi polynomial

def loiPoly(robot,t,Vmax=10):
    """ Création de la loi polynomial, à modifier il manque la vrai valeur des coeff"""
    qf = np.array([3*math.pi/4, math.pi/2])
    a0, a1, a2, a3, tf = calcCoeff(Vmax, robot,qf)
    q = np.zeros(robot.nq)
    dq = np.zeros(robot.nv)
    for i in range(robot.nq):
        q[i] = a0[i] + a1[i]*t + a2[i]*(t**2) + a3[i]*(t**3) 
        dq[i] = a1[i] + 2*a2[i]*t + 3*a3[i]*t**2
        if(t>=tf[i]):
            q[i] = a0[i] + a1[i]*tf[i] + a2[i]*(tf[i]**2) + a3[i]*(tf[i]**3) #2 dimension 
            dq[i] = a1[i] + 2*a2[i]*tf[i] + 3*a3[i]*tf[i]**2
        
    return q,dq

#Position initial du robot
def PosInit():

    return robot.q0.copy

def calcCoeff(Vmax, robot, qf):

    a0 = np.zeros(robot.nq)
    a1 = np.zeros(robot.nq)
    a2 = np.zeros(robot.nq)
    a3 = np.zeros(robot.nq)
    tf = np.zeros(robot.nq)
    DeltaQ = qf - robot.q0
    
    for i in range(robot.nq):
        tf[i] = (3/2)*(DeltaQ[i]/Vmax)
        a0[i] = robot.q0[i]         # Contrainte position initiale = qinit
        a1[i] = 0                   # Containte position initiale = 0
        a2[i] = (3*DeltaQ[i])/(tf[i]**2) 
        a3[i] = (-2*DeltaQ[i])/(tf[i]**3)

    return a0,a1,a2,a3,tf




def simulateurVerif(N,robot):

    LOCAL,WORLD = pin.ReferenceFrame.LOCAL,pin.ReferenceFrame.WORLD
    IDX = robot.model.getFrameId("tcp")
    IDB = robot.model.getFrameId("base_link")
    dt = 1e-4
    a0,a1,a2 = 0.1,0.02,-1
    X = np.zeros((N-1,3))
    t = np.zeros(N-1)
    dotXJac = np.zeros(X.shape)
    calcCoeff(10,robot,np.array([3*math.pi/4, math.pi/2]))
    for i in range(N-1):
        q,dq = loiPoly(robot,i*dt)
        #J = pin.jacobianSubtreeCenterOfMass(robot.model,robot.data,2) # essais de pin.jacobianSubtreeCenterOfMass(robot.model,robot.data,2)
        #J = pin.jacobianCenterOfMass(robot.model,robot.data,q)
        robot.forwardKinematics(q) #update joint 
        pin.updateFramePlacements(robot.model,robot.data) #update frame placement
        #J = adaptJacob(pin.computeFrameJacobian(robot.model,robot.data,q,IDX,LOCAL)) #essais de world, local comme frame de ref 
        J = adaptJacob(robot.computeFrameJacobian(q,IDX))
        X[i,:] =    adaptSituation(situationOT(robot.data.oMf[IDX]))
        dotXJac[i,:] = np.dot(J,dq)
        t[i] = i*dt
        robot.display(q)
        time.sleep(dt)
    dotXDiff = calculDotX(X,dt)
    print("shape dotX avec différences finies\t ",dotXDiff.shape)

    if PLOT:
        plt.plot(t,dotXJac[:,0],label="avec Jacobienne")
        plt.plot(t[:len(t)-1],dotXDiff[:,0],".",label="avec différences finis")
        plt.plot(t,X[:,0],label="position OT selon l'axe X")
        plt.title("vérification de la jacobienne")
        plt.plot()
        plt.legend()
        plt.show()

        


#Jacobienne

def jacobienneValidation(J,dq):
    dotX = np.dot(J,dq)
    return dotX

def calculDotX(X,dt):
    """ Calcul de Xpoint avec les différences finis, Xpoint = DeltaX/Dt en [m/s m/s rad/s ] """
    dotX = np.zeros((X.shape[0]-1,X.shape[1]))
    for i in range(X.shape[0]-1):
        dotX[i,:] = (X[i+1,:] - X[i,:])/dt
    
    return dotX 


#Passage de X dans un proportionnel
def prop(X,p):
	for i in X:
		i = i*p

def jacob(X):
    j = pin.jacobianSubtreeCenterOfMass(model,data,2)
    for i in range(3) :
        if(j[i][0] != 0.):
            X[i] *= j[i]

def mvt_Robot(X,Xd):
    Xd = Xd + X


Main = True
workingDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(workingDir)
# urdf directory path
package_path = workingDir
urdf_path = package_path + '/robots/planar_2DOF/urdf/planar_2DOF_TCP.urdf'
robot = RobotWrapper()
if Main: 
    robot = RobotWrapper.BuildFromURDF(urdf_path,package_path,verbose=True)
    robot.initViewer(loadModel=True)
    robot.display(robot.q0)
    simulateurVerif(20000,robot)

