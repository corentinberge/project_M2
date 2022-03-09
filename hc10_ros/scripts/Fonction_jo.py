import pinocchio as pin
from pinocchio.utils import *
from pinocchio.visualize import GepettoVisualizer
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np
import math



def orientationEuler(self,R):
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


def situationOT(M):
    """ cette fonction permets à partir d'un objet SE3, d'obtenir un vecteur X contenant la transaltion et la rotation de L'OT (situation de l'OT)
    avec les angles d'euler classique, M est l'objet SE3, out = [ex ey ez psi theta phi] """
    p = M.translation
    delta = orientationEuler(M.rotation)
    return np.concatenate((p,delta),axis=0)

def robotDynamic(robot,input,q,vq,aq,dt):
    """ 
    Dynamic of the robot calculator for postion/speed control 
    tau =  input + G
    tau = J't.f
    ------------------------------
    IN
    
    robot   : a RobotWrapper object needed to compute gravity torque and other parameters
    input   : input signal of the function equals to B*deltaDotQ-K*deltaQ
    q       : current joints angles values
    vq      : current joints velocities 
    aq      : current joints acceleration values 
    dt      : time step between each execution of this function
    ---------------------------------
    OUT
    q : calculated joint angles values 
    dq : calculated joint velocities values 
    aq : calculated joint acceleration values 
    f : the force exerted by the manipulator 
    system : 
            Xp = Ax + Bu
            Y = x
            with u = tau, x = [q,vq], Xp = [vq,aq]
    """


    A = pin.crba(robot.model,robot.data,q) # compute mass matrix
    H = pin.rnea(robot.model,robot.data,q,vq,np.zeros(aq.shape))  # compute dynamic drift -- Coriolis, centrifugal, gravity
    
    X = np.array([q,vq])
    Xp = np.array([vq,np.dot(np.linalg.pinv(A),(input-H))])
    X += Xp*dt

    return X[0],X[1],Xp[1]


def getdjv(robot,q,v,a,IDX):
    """this function return the product of the derivative Jacobian times the joint velocities """ 
    IDX = robot.model.getFrameId("tcp")
    linA = pin.getFrameClassicalAcceleration(robot.model,robot.data,IDX,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
    rotA = pin.getFrameClassicalAcceleration(robot.model,robot.data,IDX,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).angular
    dJv = np.hstack((linA,rotA))
    return dJv

def computedTorqueController(Xd,X,dXd,dX,ddXd,ddXn,J,A,H): 
    """
            this is the controller of the computed torque control 
            she compute the error, and return the tau ( corresponding to U(t) )
            Kp = wj²
            Kd = 2zetawj
            Xd = traj EF desired at instant t 2x1
            X =  current position of the EF 2x1
            dXd = velocities EF desired at instant t 2x1  
            dX =  current velocity of the EF 2x1 
            ddXd = acceleration of the EF desired at instant t 2x1 
            ddXn current acceleration of the EF 2x1 
            
            J the Jacobian
            A inertial matrix
            H corriolis vector 
"""
    kp=1
    kd = 2*math.sqrt(kp)
    ex = Xd-X
    edx = dXd-dX
    Jp = np.linalg.pinv(J)
    W= kp*ex + kd*edx+ddXd-ddXn
    jpw = np.dot(Jp,W)
    tau = np.dot(A,jpw) + H
    return tau


def ROS_function(robot,q,vq,aq,Xc,dXc,ddXc,dt):
    """
        this function using the joint position, velocity and acceleration, with a consigne to reach Xc, dXc, ddXc
        return the joint position, velocity and acceleration to apply to the robot
        robot : the robot built wiith pinocchio
        q : the current joint position
        vq : the current joint velocity
        aq : the current joint acceleration
        retunr the different position,velocity and acceleration to be send on the robot
    """
    
    # Derivate to obtain current acceleration
    # print(robot.data)
    # aq = (robot.data.velocity(q,vq,1)-vq)/dt
    # print(aq+"\n"+type(aq))
    # aq = [((robot.velocity(q,vq,i)- vq) / dt) for i in range(1,7)]
    # aq = (robot.velocity(q,vq,) - vq) / dt

    IDX = robot.model.getFrameId("tcp") # Change tcp for the OT desired (in the urdf file for the name)
    robot.forwardKinematics(q,vq,0*aq)
    pin.updateFramePlacement(robot.model,robot.data)
    J = pin.ComputeFrameJacobian(robot.model,robot.data,q,IDX,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    A = pin.crba(robot.model,robot.data,q) # compute mass matrix
    H = pin.rnea(robot.model,robot.data,q,vq,np.zeros(q.shape))  # compute dynamic drift -- Coriolis, centrifugal, gravity
    
    Xmeasure = situationOT(robot.data.omF[IDX])
    dXmeasure = np.dot(np.linalg(J),q)
    ddXmeasure = getdjv(robot,q,vq,aq)

    tau = computedTorqueController(Xc,Xmeasure,dXc,dXmeasure,ddXc,ddXmeasure,J,A,H)


    qnew, vqnew, aqnew = robotDynamic(robot,tau,q,vq,aq,dt)
    return qnew,vqnew,aqnew