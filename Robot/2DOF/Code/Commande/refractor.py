""" 
    This program is thre refractored prog of commande_P.Py, 
    Dim of the problem is 2X2

    Are implented :
        computed_torque_control
        Hybrid control

    those problem is for a planar2DOF Robot
    plan (O,X,Z)
"""

from cProfile import label
from numpy.linalg.linalg import det, transpose
import pinocchio as pin
from pinocchio.utils import *
from pinocchio.visualize import GepettoVisualizer
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import norm, inv, pinv
from scipy.linalg import pinv2
import time
import os

""" 
            SITUATION OT, Jacobian                  """
def situationOT(M):
    """ cette fonction permets à partir d'un objet SE3, d'obtenir un vecteur X contenant la transaltion et la rotation de L'OT (situation de l'OT)
    avec les angles d'euler classique, M est l'objet SE3, out = [ex ey ez psi theta phi] """
    p = M.translation
    return np.array([p[0],p[2]])
       

def computePlanarJacobian(robot,q,IDX):
    """
            this function compute the planar jacobian for the robot, in the reference frame 
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    """
    J = pin.computeFrameJacobian(robot.model,robot.data,q,IDX,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    #print("J",J)
    planarJ = np.vstack((J[0,:],J[2,:]))#,J[4,:]))
    #print("planar J",planarJ)
    return planarJ,J


def getdjv(robot,q,v,a):
    """this function return the product of the derivative Jacobian times the joint velocities """ 
    IDX = robot.model.getFrameId("tcp")
    #robot.forwardKinematics(q,v,0*a)
    linA = pin.getFrameClassicalAcceleration(robot.model,robot.data,IDX,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
    rotA = pin.getFrameClassicalAcceleration(robot.model,robot.data,IDX,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).angular
    #dJv = np.hstack((linA[0],linA[2],rotA[1]))
    dJv = np.hstack((linA[0],linA[2]))
    #robot.forwardKinematics(q,v,a)
    return dJv


"""                             TRAJECTOIRE                             """
def loiPendule(robot,t):
    """retourne la loi avec série de fournier """
    q = np.array([0.1*np.cos(0.1*math.pi*t), 0])
    vq = np.array([-0.01*math.pi*np.sin(0.1*math.pi*t),0])
    aq = np.array([-0.001*math.pi**2*np.cos(0.1*math.pi*t),0])
    return  q,vq,aq


def getTraj(N,robot,dt):
    """
        getTraj return a trajectory, choose a trajectory by changing loi
        by default return a polynomial law, with "P" , with other walue than "P" return a fourier law

        OUT 

        X       : OT position shape (N,3)
        dotX    : the dérivation of X (N,3)!! Warning, orientation can't be derivate
        q traj  : trajectory of joint angle
        dq traj  : trajectory of joint angle velocities
        t        : time vector of the law

    """
    X = np.zeros((2,N))
    traj_q = np.zeros((robot.nq,N))
    traj_dq = np.zeros(traj_q.shape)
    traj_ddq = np.zeros(traj_q.shape) 
    t = np.zeros(N)
    dotX = np.zeros(X.shape)
    ddX = np.zeros(X.shape)
    IDX = robot.model.getFrameId("tcp")

    for i in range(N):

        q,dq,ddq = loiPendule(robot,i*dt)
        robot.forwardKinematics(q,dq,0*ddq)
        djv = getdjv(robot,q,dq,ddq)
        pin.updateFramePlacements(robot.model,robot.data) #update frame placement  
        pJ,J = computePlanarJacobian(robot,q,IDX)
        
        #X[:,i] = np.concatenate((situationOT(robot.data.oMf[IDX]),[q[0]+q[1]]))
        X[:,i] = situationOT(robot.data.oMf[IDX])

        a = np.dot(J,dq)
        traj_dq[:,i] = dq
        traj_q[:,i] = q
        t[i] = i*dt
        dotX[:,i] = np.dot(pJ,dq)
        ddX[:,i] = djv + np.dot(pJ,ddq)
        #print("valeur 6x2",a)
        #print("valeur 2x2",dotX[:,i])

    return X,dotX,ddX,traj_q,traj_dq,traj_ddq,t


""""                        CONTROL                         """
def robotDynamicWithForce(robot,tau,q,vq,aq,f,dt):
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
    IDX = robot.model.getFrameId("tcp")
    A = pin.crba(robot.model,robot.data,q) # compute mass matrix
    H = pin.rnea(robot.model,robot.data,q,vq,np.zeros(aq.shape))  # compute dynamic drift -- Coriolis, centrifugal, gravity
    pJ,J = computePlanarJacobian(robot,q,IDX)

    X = np.array([q,vq])
    Xp = np.array([vq,np.dot(pinv(A),(tau-H))])             #-np.dot(np.transpose(pJ),f)))])
    X += Xp*dt



    q = X[0]
    vq = X[1]
    aq = Xp[1]
    
    
    robot.forwardKinematics(q,vq,0*aq) #update joint
    pin.updateFramePlacements(robot.model,robot.data) #update frame placement
    tau = pin.rnea(robot.model,robot.data,q,vq,aq)
    
    return q,vq,aq,np.dot(pinv(np.transpose(pJ)),tau),pJ,J

    



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
    
    tau = input
    X = np.array([q,vq])
    Xp = np.array([vq,np.dot(pinv(A),(tau-H))])
    X += Xp*dt

    return X[0],X[1],Xp[1]

    #COMPUTED TORQUE CONTROL

def computedTorqueController(Xd,X,dXd,dX,ddXd,ddXn,J,A,H,J6):
    """
            this is the controller of the computed torque control 

            she compute the error, and return the tau ( corresponding to U(t) )


            Kp = wj²
            Kd = 2zetawj
"""
    kp=1
    kd=0
    ex = Xd-X
    edx = dXd-dX
    #W = np.dot(kp,ex)+np.dot(kd,edx)+ddXd-ddXn
    J6p = pinv(J6) #pinv(J) => tres different
    #print(J6p)
    Jp = transpose(np.vstack((J6p[:,0],J6p[:,2])))
    #print(Jp)
    Jp = pinv(J) # uncomment for effort control 
    W= kp*ex + kd*edx+ddXd-ddXn
    jpw = np.dot(Jp,W)
    tau = np.dot(A,jpw) + H
    return tau

def run(robot):
    """
        simulate the computed torque control law


    """
    dt = 1e-3
    N = 40000
    BASE = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    IDX = robot.model.getFrameId("tcp")

    Xdesired,dXdesired,ddXdesired,traj_q,traj_dq,traj_ddq,t = getTraj(N,robot,dt)
    q = traj_q[:,0]
    vq = np.zeros(robot.nv)
    aq = np.zeros(robot.nv)
    trajX = np.zeros(Xdesired.shape)

    #robot.forwardKinematics(q) #update joints
    tau = pin.rnea(robot.model, robot.data, q, np.zeros(robot.nv), np.zeros(robot.nv)) #gravity matrix
    for i in range(N):
        robot.forwardKinematics(q,vq,0*aq) #update joint
        pin.updateFramePlacements(robot.model,robot.data) #update frame placement
        pJ,J = computePlanarJacobian(robot,q,IDX)
        A = pin.crba(robot.model,robot.data,q) # compute mass matrix
        H = pin.rnea(robot.model,robot.data,q,vq,np.zeros(aq.shape))  # compute dynamic drift -- Coriolis, centrifugal, gravity
        
        #X = np.concatenate((situationOT(robot.data.oMf[IDX])))#,[q[0]+q[1]]))
        X = situationOT(robot.data.oMf[IDX])
        Jp = pinv(pJ)
        dX = np.dot(pJ,vq)
        ddXn = getdjv(robot,q,vq,aq)
        
        trajX[:,i] = X
        q,vq,aq = robotDynamic(robot,tau,q,vq,aq,dt)
        #tau = computedTorqueController(traj_q[:,i],q,traj_dq[:,i],vq,traj_ddq[:,i],0*ddXn,np.eye(robot.model.nq),A,H) #tracking
        tau = computedTorqueController(Xdesired[:,i],X,dXdesired[:,i],dX,ddXdesired[:,i],ddXn,pJ,A,H,J) #constant position 

        robot.display(q)
        #time.sleep(1)
    plot_res(t,trajX,Xdesired)

    #HYBRID CONTROL

def PCLT(Xd,X,dXd,dX,ddXd,ddXn,Jp,A,S,J):
    """
        Position control law

        Computed with 
        
    """
    outcomputedTorqueController = np.transpose(computedTorqueController(Xd,X,dXd,dX,ddXd,ddXn,eye(X.shape[0]),eye(X.shape[0]),np.zeros(X.shape),eye(3)))
    #print("out controller avant ",outcomputedTorqueController)
    outcomputedTorqueController = np.dot(S,outcomputedTorqueController)
    
    #print("out shape = ",outcomputedTorqueController.shape)
    Jp = pinv(J)
    Jp = transpose(np.vstack((Jp[:,0],Jp[:,2])))

    outJp = np.dot(Jp,outcomputedTorqueController)

    Aout = np.dot(A,outJp)

    return  Aout

def FCL(fd,f,dX,Jt,I,dt,Kf,Kdf,I_S,KIf):
    """
        Force control law in the effort_control
        
        fd : desired force
        f : force returned by the robot
        I : integration of f
        dX : velocity of the OT

        the force f, fd are 6 by 1 vector 
        
    """

    outPID,I = controllerPID(fd-f,-dX,I,dt,Kp=Kf,Kd=Kdf,Ki=KIf)#+ fd
    out = np.dot(I_S,outPID)
    #print("deltaF = ",out)
    return np.dot(Jt,out),I


def controllerPID(delta,deltaDot,I,dt,Kp = 1,Kd = 1,Ki = 1):
    """ 
    Compute the signal of the output of the controller which can be interpreted as a PID controller
    ------------------------------
    IN 

    deltaX      : error 
    deltaDotX   : error velocity
    """
    I += delta*dt #on relaise l'intégral 
    return (np.dot(Kp,delta)+np.dot(Kd,deltaDot) + np.dot(Ki,I)),I


def effort_control(Xd,X,dXd,dX,ddXd,ddXn,fd,f,I,J,S,I_S,A,H,dt,J6):
    """
        this function compute all the effort_control 

        IN :
        
        position, velocity, acceleration of the OT desired with the force
        information from feedback position, velocity, acceleration and force
        A : the inertial matrix
        H : i forgot 
        dt : differential of time usefull for PID
        I : the Integral of the force
        S,I_S selection matrix

        OUT :
        
        tau 
    """

    Kf = 0.1
    Kdf = 0
    KIf = 0

    #if(det(J) == 0):
        #print("c'est J le probleme j'ai jurer")
    J6p = pinv(J6)
    Jp = transpose(np.vstack((J6p[:,0],J6p[:,2])))
    pclt = PCLT(Xd,X,dXd,dX,ddXd,ddXn,Jp,A,S,J6)
    #fcl,I = FCL(fd,f,dX,np.transpose(J),I,dt,Kf,Kdf,I_S,KIf)
    fcl,I = FCL(fd,f,dX,np.transpose(J),I,dt,Kf,Kdf,I_S,KIf)
    print(fcl)

    return (pclt + fcl + H),I

def run_efort_control(robot):
    """
        this function implement the efort control scheme and run it 


    """
    N = 40000
    dt = 1e-3

    IDX = robot.model.getFrameId("tcp")
    Xdesired,dXdesired,ddXdesired,traj_q,traj_dq,traj_ddq,t = getTraj(N,robot,dt)
    fd = np.array([1,1])
    q = traj_q[:,0]

    vq = np.zeros(robot.nv)
    aq = np.zeros(robot.nv)

    trajX = np.zeros(Xdesired.shape)
    trajF = np.zeros(Xdesired.shape)

    robot.forwardKinematics(q) #update joints
    pin.updateFramePlacements(robot.model,robot.data) #update frame placement
    pJ,J = computePlanarJacobian(robot,q,IDX)

    S = np.eye(2)
    S = np.diag([1,0])
    I_S = np.eye(S.shape[0])-S

    tau = pin.rnea(robot.model, robot.data, q, np.zeros(robot.nv), np.zeros(robot.nv)) #gravity matrix
    f = np.dot(np.transpose(pinv(pJ)),tau)
    f = np.zeros(robot.nq)
    I=0
    print(I_S)
    for i in range(N):
        
        X = situationOT(robot.data.oMf[IDX])
        trajX[:,i] = X

        q,vq,aq,f,pJ,J6 = robotDynamicWithForce(robot,tau,q,vq,aq,f,dt)
    
        A = pin.crba(robot.model,robot.data,q)
        H = pin.rnea(robot.model,robot.data,q,vq,np.zeros(aq.shape))
        dX = np.dot(pJ,vq)
        ddX = getdjv(robot,q,vq,aq)

        #print("postion error before controller",norm(adaptSituation(Xdesired[:,i])-X))

        #input,I = effort_control(adaptSituation(Xdesired[:,Id]),X,0*adaptSituation(dXdesired[:,0]),dX,0*adaptSituation(ddXdesired[:,0]),adaptSituation(ddX),fd,f,I,J,S,I_S,A,H,dt) #constant position
        tau,I = effort_control(Xdesired[:,i],X,dXdesired[:,i],dX,ddXdesired[:,i],ddX,fd,f,I,pJ,S,I_S,A,H,dt,J) # tracking 
        #tau = PCLT(Xdesired[:,i],X,dXdesired[:,i],dX,ddXdesired[:,i],ddX,pinv(pJ),A,S,J6)
        #tau += H
        trajF[:,i] = f
        #time.sleep(dt)
    plot_res(t,trajX,Xdesired)
    plt.figure()
    plt.title("Force sur Z")
    plt.plot(t,trajF)
    plt.show()





"""                         LOADING URDF FILE                   """
def getRobot():
    """ load urdf file  """
    package_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/Modeles/'
    urdf_path = package_path + 'planar_2DOF/URDF/planar_2DOF_TCP.urdf'
    robot = RobotWrapper.BuildFromURDF(urdf_path,package_path,verbose=True)
    return robot

def init_display(robot):
    """ load display   """
    robot.initViewer(loadModel=True)
    robot.display(robot.q0)

"""                     PLOT                        """
def plot_res(t,trajX,Xdesired):
    plt.figure()
    plt.title("trajectoire sur X")
    plt.plot(t,trajX[0,:],label="trajectoire mesurer")
    plt.plot(t,Xdesired[0,:],'r--',label="trajectoire à suivre")
    plt.figure()
    plt.title("trajectoire sur Z")
    plt.plot(t,trajX[1,:],label="trajectoire mesurer")
    plt.plot(t,Xdesired[1,:],'r--',label="trajectoire à suivre")
    plt.legend()
    plt.show()

"""         TEST            """

def test(robot):
    dt = 1e-3
    N = 400000
    Xdesired,dXdesired,ddXdesired,traj_q,traj_dq,traj_ddq,t = getTraj(N,robot,dt)

    ddXverif = calculDotX(dXdesired,dt)
    dXverif = calculDotX(Xdesired,dt)
    #print(t[0:len(t)-1].shape)
    plt.figure()
    plt.title("verif vit")
    plt.plot(t[0:len(t)-1],dXverif[0,:],'r--',label="diff finis",linewidth=10)
    plt.plot(t,dXdesired[0,:],label="vit calculer")
    plt.legend()
    plt.figure()
    plt.title("verif acc")
    plt.plot(t[0:len(t)-1],ddXverif[0,:],'r--',label="diff finis",linewidth=5)
    plt.plot(t,ddXdesired[0,:],label="vit calculer")
    plt.legend()
    plt.show()

def calculDotX(X,dt):
    """ Calcul de Xpoint avec les différences finis, Xpoint = DeltaX/Dt en [m/s m/s rad/s ] """
    dotX = np.zeros((X.shape[0],X.shape[1]-1))
    for i in range(X.shape[1]-1):
        dotX[:,i] = (X[:,i+1] - X[:,i])/dt
    #print("shape : ",dotX.shape)
    return dotX 


if __name__ == '__main__':
    N = 10
    dt = 1e-4
    robot = getRobot()
    init_display(robot)

    #test(robot)
    #run(robot)
    run_efort_control(robot)