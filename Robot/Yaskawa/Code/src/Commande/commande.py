""" 

    Simulation de loi de commande pour le yaskawa 

"""
import os
import math
import time
import pinocchio as pin
from pinocchio.utils import *
from pinocchio.visualize import GepettoVisualizer
from pinocchio.robot_wrapper import RobotWrapper


"""
    OT situation
"""

def situationOT(M):
    """ cette fonction permets à partir d'un objet SE3, d'obtenir un vecteur X contenant la transaltion et la rotation de L'OT (situation de l'OT)
    avec les angles d'euler classique, M est l'objet SE3, out = [ex ey ez psi theta phi] """
    p = M.translation
    delta = orientationEuler(M.rotation)
    return np.concatenate((p,delta),axis=0)
        

def orientationEuler(R):
    """ Renvois l'orientation selon la valeurs des angles d'euler  
    prend une matrice de rotation 3x3 en entrée"""
    if(abs(R[2,2]) != 1):
        psi = math.atan2(R[0,2],-R[1,2])
        theta = math.acos(R[2,2])
        phi = math.atan2(R[2,0],R[2,1])
    else : # attention psi et phi ne sont pas définis ici phi = 2*psi => évite la division par 0 
        #print("attention psi et phi ne sont pas définis ici ils seront pris égaux")
        a = math.atan2(R[0,1],R[0,0])
        psi = a/(1-2*R[2,2])
        theta = math.pi*(1-R[2,2])/2
        phi = 2*psi
    return np.array([psi,theta,phi])


"""
    Movement Generation
"""

def getTraj(N,robot,IDX,dt,law='P',V=10,qf = np.array([math.pi/3,0.001,math.pi,-math.pi/2,-0.001,math.pi/4]) ):
    """
        getTraj return a trajectory, choose a trajectory by changing loi
        by default return a polynomial law, with "P" , with other walue than "P" return a fourier law

        In :
            N Number of point 
            IDX the frame of the end effector 
            law the law you want Fourier or Polynomial
            V 
        OUT 

        X       : OT position shape (N,3)
        dotX    : the dérivation of X (N,3)!! Warning, orientation can't be derivate
        q traj  : trajectory of joint angle
        dq traj  : trajectory of joint angle velocities
        t        : time vector of the law

    """
    X = np.zeros((N,6))
    traj_q = np.zeros((N,robot.nq))
    traj_dq = np.zeros(traj_q.shape) 
    t = np.zeros(N)
    dotX = np.zeros(X.shape)
    for i in range(N):
        if(law == 'P'):
            q,dq = loiPoly(robot,i*dt,qf,Vmax=V)
        else:
            q,dq = loiPendule(robot,i*dt)
        robot.forwardKinematics(q) #update joint 
        pin.updateFramePlacements(robot.model,robot.data) #update frame placement  
        J = pin.computeFrameJacobian(robot.model,robot.data,q,IDX,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        X[i,:] =  situationOT(robot.data.oMf[IDX])
        traj_dq[i,:] = dq
        traj_q[i,:] = q
        t[i] = i*dt
        dotX[i,:] = np.dot(J,dq)
    return X,dotX,traj_q,traj_dq,t
        

def loiPoly(robot,t,qf,Vmax=10):
    """
        Polynomial law,
        
        IN : 
            robot object, which contain data, etc....
            t, time
            Vmax Velocities maximum of the robot Joint

        OUT :
            q joint angle at time t
            dq joint velocities at time t
    """
    a0, a1, a2, a3, tf = calcCoeff(Vmax, robot,qf)
    if(t == 0):
        a0, a1, a2, a3, tf = calcCoeff(Vmax, robot,qf)
        print("a0 : \t",a0)
        print("a1 : \t",a1)
        print("a2 : \t",a2)
        print("a3 : \t",a3)
        print("tf : \t",tf)
    q = np.zeros(robot.nq)
    dq = np.zeros(robot.nv)
    for i in range(robot.nq):
        q[i] = a0[i] + a1[i]*t + a2[i]*(t**2) + a3[i]*(t**3) 
        dq[i] = a1[i] + 2*a2[i]*t + 3*a3[i]*t**2
        if(t>=tf[i]):
            q[i] = a0[i] + a1[i]*tf[i] + a2[i]*(tf[i]**2) + a3[i]*(tf[i]**3) #2 dimension 
            dq[i] = a1[i] + 2*a2[i]*tf[i] + 3*a3[i]*tf[i]**2
        
    return q,dq

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

def loiPendule(robot,t):
    """ 
       fourier trajectory for a robot

       IN : 
            robot object
            t time 

        OUT :
            q joint angle at time t
            dq joint velocities at time t
     """
    q = []
    dq = []
    for i in range(robot.nq):
        q.append(0.5*np.cos(2*math.pi*t))
        dq.append(-1*math.pi*np.sin(2*math.pi*t))
    return np.array(q),np.array(dq)

"""
    Simulator
"""

def simulator(robot):
    IDX = robot.model.getFrameId("tool0")
    dt = 1e-3
    N = 20000
    Xc,dotXc,qc,dqc,t = getTraj(N,robot,IDX,dt,law='R',V=1)

    for i in range(N):
        robot.forwardKinematics(qc[i])
        pin.updateFramePlacements(robot.model,robot.data)
        robot.display(qc[i])
        time.sleep(dt)

""" 
    Control Law
"""
if __name__ == "__main__":
    workingDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    workingDir += '/Modeles'
    package_dir = workingDir
    urdf_file = workingDir + '/motoman_hc10_support/urdf/hc10dt.urdf'
    robot = RobotWrapper.BuildFromURDF(urdf_file,package_dir,verbose=True)
    
    robot.initViewer(loadModel=True)
    robot.display(robot.q0)
    simulator(robot)
    

    

        

    