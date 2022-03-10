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
    return planarJ


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
    q = np.array([0.1*np.cos(math.pi*t), 0]) + np.array([0.5,-1])
    vq = np.array([-0.1*math.pi*np.sin(math.pi*t),0])
    aq = np.array([-0.1*math.pi**2*np.cos(math.pi*t),0])
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
        #robot.display(q)
        J = computePlanarJacobian(robot,q,IDX)
        
        X[:,i] = situationOT(robot.data.oMf[IDX])

        a = np.dot(J,dq)
        traj_dq[:,i] = dq
        traj_q[:,i] = q
        t[i] = i*dt
        dotX[:,i] = np.dot(J,dq)
        ddX[:,i] = djv + np.dot(J,ddq)


    return X,dotX,ddX,traj_q,traj_dq,traj_ddq,t


""""                        CONTROL                         """
    



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
    Xp = np.array([vq,np.dot(pinv(A),(input-H))])
    X += Xp*dt

    return X[0],X[1],Xp[1]

    #COMPUTED TORQUE CONTROL

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
            
            J planar Jacobian size 2x2
            A inertial matrix
            H corriolis vector 


"""
    kp=1
    kd = 2*math.sqrt(kp)
    ex = Xd-X
    edx = dXd-dX
    #print("det J ",np.linalg.det(J))
    Jp = pinv(J)
    W= kp*ex + kd*edx+ddXd-ddXn
    jpw = np.dot(Jp,W)
    tau = np.dot(A,jpw) + H
    return tau

def run(robot):
    """
        simulate the computed torque control law


    """

    """       Initialisation            """
    dt = 1e-4
    N = 100000
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
        #print("X",X)
        Jp = pinv(pJ)
        dX = np.dot(pJ,vq)
        ddXn = getdjv(robot,q,vq,aq)
        
        trajX[:,i] = X
        q,vq,aq = robotDynamic(robot,tau,q,vq,aq,dt)
        tau = computedTorqueController(Xdesired[:,i],X,dXdesired[:,i],dX,ddXdesired[:,i],ddXn,pJ,A,H) #tracking
        #tau = computedTorqueController(Xdesired[:,1000],X,0*dXdesired[:,1000],dX,0*ddXdesired[:,1000],ddXn,pJ,A,H) #constant position 
        robot.display(q)
        #time.sleep(1)

    #plot_res(t,trajX,np.array([Xdesired[0,1000]*np.ones(Xdesired.shape[1]),Xdesired[1,1000]*np.ones(Xdesired.shape[1])])) #cnstant position plot
    plot_res(t,trajX,Xdesired) #tracking plot 

    #HYBRID CONTROL

def PCLT(Xd,X,dXd,dX,ddXd,ddXn,J,A,S):
    """
        Position control law

        Computed with 
        
    """
    kp=1
    kd = 2*math.sqrt(kp)
    ex = Xd-X
    edx = dXd-dX
    #print("det J ",np.linalg.det(J))
    Jp = pinv(J)
    W= np.dot(S,kp*ex + kd*edx+ddXd-ddXn)
    jpw = np.dot(Jp,W)
    tau = np.dot(A,jpw)

    return  tau

def FCL(fd,f,dX,J,I,dt,I_S):
    """
        Force control law in the effort_control
        
        fd : desired force
        f : force returned by the robot
        I : integration of f
        dX : velocity of the OT

        the force f, fd are 6 by 1 vector 
        
    """
    kf = 0.1
    kfd = 0
    kif = 0.0
    ef = fd-f
    I += ef*dt
    #print("error on the force : ",ef)
    a = np.dot(I_S,np.dot(kf,ef)+np.dot(kif,ef)-np.dot(kfd,dX)+1*fd) #intermediate signal
    #print("shape before Jt",a.shape)
    return np.dot(a,np.transpose(J)),I # a la base en Jt,a



def effort_control(Xd,X,dXd,dX,ddXd,ddXn,fd,f,I,J,S,I_S,A,H,dt):
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

    tau_pos = PCLT(Xd,X,dXd,dX,ddXd,ddXn,J,A,S) + H
    tau_f,I = FCL(fd,f,dX,J,I,dt,I_S)
   
    return (tau_pos + tau_f),I # return of tau_f for plot

def run_efort_control(robot):
    """
        this function implement the efort control scheme and run it 


    """
    """             INITIALISATION              """
    """ creating final and initial position """
    qi = np.array([np.radians(-46),np.radians(-46)])
    qf = np.array([np.radians(-45),np.radians(-45)])
    IDX = robot.model.getFrameId("tcp")
    robot.forwardKinematics(qf) #update joints
    pin.updateFramePlacements(robot.model,robot.data) #update frame placement
    Xf = situationOT(robot.data.oMf[IDX])
    robot.forwardKinematics(qi) #update joints
    pin.updateFramePlacements(robot.model,robot.data) #update frame placement
    """ Initialisation of robot constant"""
    """ contact constant """
    Pz0 = 1.28
    stiffness = 1
    damping = 0.1
    """ Simulation Data """
    N = 60000
    dt = 1e-3
    constant_position = 10000
    IDX = robot.model.getFrameId("tcp")

    fd = np.array([0,0.01])
    q = qi
    vq = np.zeros(robot.nv)
    aq = np.zeros(robot.nv)
    t = np.zeros(N)
    trajX = np.zeros((2,N))
    trajF = np.zeros((2,N))
    tau_f_traj = np.zeros((2,N))
    f = np.zeros((2,1))
    robot.forwardKinematics(q) #update joints
    pin.updateFramePlacements(robot.model,robot.data) #update frame placement
    J = computePlanarJacobian(robot,q,IDX)
    S = np.diag([1,0]) #on commande en position au début
    I_S = np.eye(S.shape[0])-S
    tau = pin.rnea(robot.model, robot.data, q, np.zeros(robot.nv), np.zeros(robot.nv)) #initial tau
    I=0
    Xdesired = Xf
    
    """         SIMULATION          """
    for i in range(N):
        qold = q.copy()
        t[i] = i*dt
        robot.display(q)
        q,vq,aq = robotDynamic(robot,tau,q,vq,aq,dt)

        robot.forwardKinematics(q) #update joints
        pin.updateFramePlacements(robot.model,robot.data) #update frame placement
        J = computePlanarJacobian(robot,q,IDX)
        X = situationOT(robot.data.oMf[IDX])
        
        trajX[:,i] = X
        A = pin.crba(robot.model,robot.data,q)
        H = pin.rnea(robot.model,robot.data,q,vq,np.zeros(aq.shape))
        dX = np.dot(J,vq)
        ddX = getdjv(robot,q,vq,aq)
        #print("position sur Z",X[1])
        #print("depassement ",Pz0 - X[1])
        fx = 0
        fz = stiffness*(X[1] - Pz0) + damping*dX[1]# force exerted on the static surface Pz0
        if(X[1] - Pz0>0):
            q = qold
        f = np.array([fx,fz])
        


        #tau,I = effort_control(Xdesired[:,i],X,dXdesired[:,i],dX,ddXdesired[:,i],ddX,fd,f,I,J,S,I_S,A,H,dt) # tracking 
        #tau,I,tau_f_traj[:,i] = effort_control(Xdesired[:,constant_position],X,0*dXdesired[:,constant_position],dX,0*ddXdesired[:,constant_position],ddX,fd,f,I,J,S,I_S,A,H,dt) # constant position
        tau,I = effort_control(Xdesired,X,np.zeros(X.shape),dX,np.zeros(X.shape),ddX,fd,f,I,J,S,I_S,A,H,dt)
        
        tau_f_traj[:,i] = tau
        trajF[:,i] = f
        #time.sleep(dt)



    #plot_res(t,trajX,Xdesired) #plot tracking
    plt.figure()
    plt.plot(t,trajX[0,:],label="sur X")
    plt.plot(t,Xf[0]*np.ones(t.shape),'r--',label="consigne sur X")
    plt.legend()
    plt.figure()
    plt.plot(t,trajX[1,:],label="sur Z")
    plt.plot(t,Xf[1]*np.ones(t.shape),'r--',label="consigne sur Z")
    plt.legend()
    plt.figure()
    plt.title("Force sur X")
    plt.plot(t,trajF[0,:])
    plt.figure()
    plt.title("Force sur Z")
    plt.plot(t,trajF[1,:])
    plt.ylabel('en N')
    plt.figure()
    plt.title("couple tau en q1 ")
    plt.plot(t,tau_f_traj[0,:],label='en q1')
    plt.legend()
    plt.ylabel('en N.m')
    plt.figure()
    plt.title("couple tau en q2")
    plt.plot(t,tau_f_traj[1,:],label='en q2')
    plt.legend()
    plt.ylabel('en N.m')
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
    plt.plot(t,trajX[0,:],label="trajectoire mesurée")
    plt.plot(t,Xdesired[0,:],'r--',label="trajectoire à suivre")
    plt.xlabel("seconde")
    plt.legend()
    plt.figure()
    plt.title("trajectoire sur Z")
    plt.plot(t,trajX[1,:],label="trajectoire mesurée")
    plt.plot(t,Xdesired[1,:],'r--',label="trajectoire à suivre")
    plt.xlabel("seconde")
    plt.legend()
    #plt.show()

"""         TEST            """
    
    
    
    

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

    #run(robot)
    run_efort_control(robot)