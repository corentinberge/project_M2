<<<<<<< HEAD
from numpy.linalg.linalg import transpose
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


def adaptSituation(X,q):
    """ cette fonction permets d'adapter la situation de l'organe terminal au plan, ressort un vecteur 3,3
    on enleve ey car il n'y a pas de translation selon ey, on enleve psi et phi car rotation constantes """
    nX = np.array([X[0],X[2],q[0]+q[1]])#,X[4]]) # on enleve les colonnes 2,3 et 5 donc on enlève les lignes 2,3 et 5 de la jacobienne et on enleve l'indice 4 car on travail en position pour l'insatnt
    return nX


def adaptJacob(J):
    """ cette fonction permets d'adapter la jacobienne, on enlève les lignes 1,3 et 5, ON travai avec la position donc on enlève l'indice 4 (on la garde si on veut l'orientation) """
    nJ = np.array([J[0,:],J[2,:],J[4,:]])
    return nJ

def getTraj(N,robot,IDX,dt,loi='P',V=10):
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
    a0,a1,a2 = 0,1,2
    X = np.zeros((N,3))
    traj_q = np.zeros((N,robot.nq))
    traj_dq = np.zeros(traj_q.shape) 
    t = np.zeros(N)
    dotX = np.zeros(X.shape)
    for i in range(N):
        if(loi == 'P'):
            q,dq = loiPoly(robot,i*dt,Vmax=V)
        else:
            q,dq = loiPendule(i*dt)
        robot.forwardKinematics(q) #update joint 
        pin.updateFramePlacements(robot.model,robot.data) #update frame placement  
        J = adaptJacob(pin.computeFrameJacobian(robot.model,robot.data,q,IDX,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        X[i,:] =  adaptSituation(situationOT(robot.data.oMf[IDX]),q)
        traj_dq[i,:] = dq
        traj_q[i,:] = q
        t[i] = i*dt
        dotX[i,:] = np.dot(J,dq)
    print("shape dotX : \t",dotX.shape)
   # plt.plot(t,traj_dq,"label vrai valeur de dq")
    #print("shape dotX",dotX.shape)
    #print("shape X",X.shape)
    return X,dotX,traj_q,traj_dq,t
        
#Loi polynomial

def loiPoly(robot,t,Vmax=10):
    """ Création de la loi polynomial, à modifier il manque la vrai valeur des coeff"""
    qf = np.array([math.pi/3, +6*math.pi/4])
    a0, a1, a2, a3, tf = calcCoeff(Vmax, robot,qf) # a modifier pour le calculer seulement a la premiere itération
    #print("t \t",t)
    if(t == 0):
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

#Position initial du robot
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

def loiPendule(t):
    """retourne la loi avec série de fournier """
    return np.array([0.1*np.cos(2*math.pi*t),0.5*np.cos(2*math.pi*t)]),np.array([-0.2*math.pi*np.sin(2*math.pi*t),-1*math.pi*np.sin(2*math.pi*t)])


def simulateurVerif(N,robot):
    """ Simutalteur de vérification de la jacobienne """
    BASE = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    IDX = robot.model.getFrameId("tcp")
    dt = 1e-2
    X = np.zeros((N,3))
    t = np.zeros(N)
    dotXJac = np.zeros(X.shape)
    for i in range(N):
        #q,dq = loiPendule(i*dt)
        q,dq = loiPoly(robot,i*dt,Vmax=4)
        robot.forwardKinematics(q) #update joint 
        pin.updateFramePlacements(robot.model,robot.data) #update frame placement
        if(i*dt == 0.5):
            print("qpoint = \t",dq)
            print("q \t",q)
            print("J \t",J)
        J = adaptJacob(pin.computeFrameJacobian(robot.model,robot.data,q,IDX,BASE)) #essais de world, local comme frame de ref 
        #J = adaptJacob(robot.computeFrameJacobian(q,IDX))
        X[i,:] =    adaptSituation(situationOT(robot.data.oMf[IDX]),q)
        dotXJac[i,:] = np.dot(J,dq)
        t[i] = i*dt
        robot.display(q)
        time.sleep(dt)
    dotXDiff = calculDotX(X,dt)
    print("shape dotX avec différences finies\t ",dotXDiff.shape)
    print(X[N-1,:])
    if PLOT:
        plt.plot(t,dotXJac[:,0],label="avec Jacobienne")
        plt.plot(t[:len(t)-1],dotXDiff[:,0],".",label="avec différences finis")
        plt.plot(t,X[:,0],label="position OT selon l'axe X")
        plt.title("position et dérivé situation OT selon axe x")
        plt.ylabel("m ou m/s")
        plt.xlabel("seconde")
        plt.legend()
        plt.savefig("jacobian_checkx")
        plt.figure()
        plt.plot(t,X[:,1],label="position selon axe z en m")
        plt.plot(t,dotXJac[:,1],label="avec Jacobienne en m/s")
        plt.plot(t[:len(t)-1],dotXDiff[:,1],".",label="avec différences finis en m/s")
        plt.title("position et dérivé situation OT selon axe z")
        plt.ylabel("m ou m/s")
        plt.xlabel("seconde")
        plt.legend()
        plt.savefig("jacobian_check")

def computeError(Xconsigne,Xactuel,dotXconsigne,dotXactuel):
    """ Renvois l'erreur de la situation de l'organe terminal et l'erreur de ça vitesse"""
    epsX = (Xconsigne-Xactuel)
    epsdotX = (dotXconsigne-dotXactuel)
    return epsX,epsdotX

def simuLoiCommande(robot):
    BASE = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    IDX = robot.model.getFrameId("tcp")
    dt = 1e-2
    N = 300
    Xc,dotXc,qc,dqc,t  = getTraj(N,robot,IDX,dt,loi='R',V=5) # on récupére la consigne 
    q = robot.q0 
    dq = np.zeros(robot.nq) #accélération des joint à l'instant initial
    traj_OT = np.zeros(Xc.shape)
    traj_dotOT = np.zeros(dotXc.shape)
    deltaX = np.zeros(Xc.shape[1])
    t = np.zeros(N)

    for i in range(N):
        robot.forwardKinematics(q) #update joint 
        pin.updateFramePlacements(robot.model,robot.data) #update frame placement
        J = adaptJacob(pin.computeFrameJacobian(robot.model,robot.data,q,IDX,BASE)) #calcul de la jacobienne
        X = adaptSituation(situationOT(robot.data.oMf[IDX]),q) #deltaX
        dotX = np.dot(J,dq)
        #print(dotX)
        deltaX,deltaDotX = computeError(Xc[i,:],X,dotXc[i,:],dotX)
        q = loiCommande1(deltaX,1,J,q)
        #print("q\t",q)
        traj_OT[i,:] = X
        traj_dotOT[i,:] = dotX
        t[i] = i*dt
        robot.display(q)
        time.sleep(dt)

    robot.forwardKinematics(q) #update joint 
    pin.updateFramePlacements(robot.model,robot.data) #update frame placement
    X= adaptSituation(situationOT(robot.data.oMf[IDX]),q)
    print("position x OT" + "\tposition consigne",X[0],Xc[N-1,:][0])
    print("position x OT" + "\tposition consigne",X[1],Xc[N-1,:][1])
    print("orientation OT" + "\torientation consigne",X[2],Xc[N-1,:][2])
    if PLOT: 
        plt.plot(t,traj_OT[:,0],label="position OT selon axe x")
        plt.plot(t,traj_OT[:,1],label="position OT selon axe z")
        #plt.plot(t,traj_OT[:,2],label="orientation OT")
        plt.plot(t,Xc[:,0],label="position consigne selon axe x")
        plt.plot(t,Xc[:,1],label="position consigne selon axe z")
        #plt.plot(t,Xc[:,2],label="orientation consigne")
        plt.legend()
        plt.savefig("asservissementPposition")
        plt.figure()
        plt.plot(t,dotXc[:,0],label="vitesse consigne axe x")
        plt.plot(t,traj_dotOT[:,0],label="vitesse OT axe x")
        plt.plot(t,dotXc[:,1],label="vitesse consigne axe y")
        plt.plot(t,traj_dotOT[:,1],label="vitesse OT axe y")
        #plt.plot(t,dotXc[:,2],label="vitesse angulaire de la consigne")
        #plt.plot(t,traj_dotOT[:,2],label="vitesse angulaire OT")
        plt.legend()
        plt.savefig("asservissementPVit")
        #plt.show()
    
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

def loiCommande1(deltaX,Kp,J,q):
    """ calcul du la boucle ouverte """
    print("q\t",q)
    deltaQ = np.dot(pinv(J),Kp*deltaX)
    q = moveRobot(q,deltaQ)
    #print("q\t",q)
    return q 

def loiCommande2(deltaDotX,K,J,q):
    """ Commande Seulement en vitesse """
    deltaDotQ = K*np.dot(pinv(J),deltaDotX)

    #ça ce passe dans le robot 
    deltaQ = deltaDotQ*1e-2 
    print("deltaQ\t ",deltaQ)
    q = moveRobot(q,deltaQ)
    #q = rob(q,deltaQ,1e-2,robot)
    return q,deltaQ

def moveRobot(q,deltaQ):
    """ fonction qui donne le mouvement du robot"""
    q += deltaQ
    return q

def robotDynamic(robot,input,q,vq,aq,dt):
    """ 
    Dynamic of the robot calculator for postion/speed control 
    tau =  input + G
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



    system : 
            Xp = Ax + Bu
            Y = x
            with u = tau, x = [q,vq], Xp = [vq,aq]
    """
    G = pin.rnea(robot.model, robot.data, q, np.zeros(robot.nv), np.zeros(robot.nv)) #gravity matrix
    A = pin.crba(robot.model,robot.data,q) # compute mass matrix
    H = pin.rnea(robot.model,robot.data,q,vq,aq)  # compute dynamic drift -- Coriolis, centrifugal, gravity
    tau = input+G
    X = np.array([q,vq])
    Xp = np.array([vq,np.dot(pinv(A),(tau-H))])
    X += Xp*dt

    return X[0],X[1],Xp[0]

def simulator(robot,espace="OT"):
    """
    
    """
    N = 500
    BASE = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    IDX = robot.model.getFrameId("tcp")
    dt = 1e-2
    Xc,dotXc,qc,dqc,t = getTraj(N,robot,IDX,dt,loi='R',V=5)
    q = robot.q0    
    dq = np.zeros(robot.nv) #accélération des joint à l'instant initial
    ddq = np.zeros(robot.nv)
    traj_OT = np.zeros(Xc.shape)
    traj_dotOT = np.zeros(dotXc.shape)
    I = 0
    for i in range(N):
        
        robot.forwardKinematics(q) #update joint 
        pin.updateFramePlacements(robot.model,robot.data) #update frame placement
        J = pin.computeFrameJacobian(robot.model,robot.data,q,IDX,BASE)
        X = situationOT(robot.data.oMf[IDX])
        J = adaptJacob(J) # adadptedJ
        X = adaptSituation(X,q) #adaptedX
        dotX = np.dot(J,dq)
        print("derivative Jacobian \n",robot.data.dJ)
        print("\n\n\n")
        # adapaptation en prenant en compte le plan 
        if espace == "OT":
            deltaX,deltaDotX = computeError(Xc[i,:],X,dotXc[i,:],dotX) #dotXc[4,:] 
            #Kp,Kd,Ki = calcGain(robot,3.3)

            outController,I = controller(deltaX,deltaDotX,I,dt,Kp=62.5,Kd=20,Ki=0.2)
            Jplus = pinv(J)
            Jt = transpose(J)
            inRobot = np.dot(Jt,outController)
        elif espace == "Joint":
            deltaQ,deltaDotQ = computeError(qc[N-1,:],q,0,dq)
            Kp,Kd,Ki = calcGain(robot,4)
            outController,I = controller(deltaQ,deltaDotQ,I,dt,Kp,Kd,Ki)
            inRobot = outController

        
        
        #print(q)
        #print(dq)
        #print("Joint acceleration =\t",ddq)
        #print("J \n",J )

        q,dq,ddq = robotDynamic(robot,inRobot,q,dq,ddq,dt)
        traj_OT[i,:] = X
        traj_dotOT[i,:] = dotX
        # Display of the robot
        robot.display(q)
        time.sleep(dt)
    
    print("position x OT" + "\tposition consigne",X[0],Xc[N-1][0])#Xc[N-1[0],:])
    print("position y OT" + "\tposition consigne",X[1],Xc[N-1][1])#Xc[N-1,:][1])
    print("orientation OT" + "\torientation consigne",X[2],Xc[N-1][2])#Xc[N-1,:][2])
    if PLOT: 
        plt.plot(t,traj_OT[:,0],label="position OT selon axe x")
        plt.plot(t,traj_OT[:,1],label="position OT selon axe y")    
        #plt.plot(t,traj_OT[:,2],label="orientation OT")
        plt.plot(t,Xc[:,0],label="position consigne selon axe x")
        plt.plot(t,Xc[:,1],label="position consigne selon axe y")
        #plt.plot(t,Xc[:,2],label="orientation consigne")
        plt.legend()
        plt.figure()
        #plt.plot(t,dotXc[:,0],label="vitesse consigne axe x")
        plt.plot(t,traj_dotOT[:,0],label="vitesse OT axe x")
        #plt.plot(t,dotXc[:,1],label="vitesse consigne axe y")
        plt.plot(t,traj_dotOT[:,1],label="vitesse OT axe y")
        #plt.plot(t,dotXc[:,2],label="vitesse angulaire de la consigne")
        #plt.plot(t,traj_dotOT[:,2],label="vitesse angulaire OT")
        plt.legend()
        plt.show()


def controller(delta,deltaDot,I,dt,Kp = 1,Kd = 1,Ki = 1):
    """ 
    Compute the signal of the output of the controller which can be interpreted as a PD controller
    ------------------------------
    IN 

    deltaX      : error 
    deltaDotX   : error velocity
    """
    #print("KP \t",Kp)
    I += delta*dt
    return (np.dot(Kp,delta)+np.dot(Kd,deltaDot) + np.dot(Ki,I)),I

def calcGain(robot,w):
    A = robot.data.M
    f = robot.model.friction
    Kp = []
    Kd = []
    KI = []
    #print("\n\n\n Friction \n\n\n",f)
    for i in range(robot.model.nq):
        Kp.append(3*A[i,i]*w**2)
        Kd.append(3*A[i,i]*w - f[i])
        KI.append(A[i,i]*w**3)

    Kp = np.diag(Kp)
    Kd = np.diag(Kd)
    KI = np.diag(KI)
#    print("gain Intégral \t",KI)
#    print("gain dériver \t",Kd)
#    print("gain prop \t",Kp)
    return Kp,Kd,KI
    
def computeJacobianDerivative():
    



Main = True
package_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/Modeles/'
urdf_path = package_path + 'planar_2DOF/URDF/planar_2DOF_TCP.urdf'

robot = RobotWrapper()
if __name__ == '__main__':
    robot = RobotWrapper.BuildFromURDF(urdf_path,package_path,verbose=True)
    robot.initViewer(loadModel=True)
    robot.display(robot.q0)
    #simulateurVerif(300,robot)
    simulator(robot)
    #simuLoiCommande(robot)
    #simuLoiCommande(robot)
=======
from numpy.core.fromnumeric import shape
from numpy.core.numeric import ones
from numpy.lib.index_tricks import ix_
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

MAIN = True #Execution du main
PLOT = True #pour avoir les PLOTS


def situationOT(M):
    """ cette fonction permets à partir d'un objet SE3, d'obtenir un vecteur X contenant la transaltion et la rotation de L'OT (situation de l'OT)
    avec les angles d'euler classique, M est l'objet SE3, out = [ex ey ez psi theta phi] """
    p = M.translation
    #delta = orientationEuler(M.rotation) #à decommenter a terme 
    delta = np.zeros(3)
    return np.concatenate((p,delta),axis=0)
       

def orientationEuler(R):
    """ Renvois l'orientation selon la valeurs des angles d'euler  
    prend une matrice de rotation 3x3 en entrée"""
    print("R22 = ",R[2,2])
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


def adaptSituation(X,q):
    """ cette fonction permets d'adapter la situation de l'organe terminal au plan, ressort un vecteur 3,3
    on enleve ey car il n'y a pas de translation selon ey, on enleve psi et phi car rotation constantes """
    nX = np.array([X[0],X[2],q[0]+q[1]])#,X[4]]) # on enleve les colonnes 2,3 et 5 donc on enlève les lignes 2,3 et 5 de la jacobienne et on enleve l'indice 4 car on travail en position pour l'insatnt
    return nX
    
def adaptPlanarRotation(R):
    """
        Adapt the rotation matrix, with respect of the Plan(0,x,z)
    """
    new_R = np.array([[R[0,0],R[0,2]],[R[2,0],R[2,2]]])
    return new_R

def adaptJacob(J):
    """ cette fonction permets d'adapter la jacobienne, on enlève les lignes 1,3 et 5, ON travai avec la position donc on enlève l'indice 4 (on la garde si on veut l'orientation) """
    nJ = np.array([J[0,:],J[2,:],J[4,:]])
    return nJ

def getTraj(N,robot,IDX,dt,loi='P',V=10):
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
    a0,a1,a2 = 0,1,2
    X = np.zeros((6,N))
    traj_q = np.zeros((robot.nq,N))
    traj_dq = np.zeros(traj_q.shape)
    traj_ddq = np.zeros(traj_q.shape) 
    t = np.zeros(N)
    dotX = np.zeros(X.shape)
    ddX = np.zeros(X.shape)
    IDX = robot.model.getFrameId("tcp")

    for i in range(N):
        if(loi == 'P'):
            q,dq,ddq = loiPoly(robot,i*dt,Vmax=V)
        else:
            q,dq,ddq = loiPendule(robot,i*dt)
        robot.forwardKinematics(q,dq,0*ddq)
        dJv = np.hstack( (pin.getFrameAcceleration(robot.model,robot.data,IDX,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear ,pin.getFrameAcceleration(robot.model,robot.data,IDX,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).angular)) 
        robot.forwardKinematics(q,dq,ddq) #update joint 
        pin.updateFramePlacements(robot.model,robot.data) #update frame placement  
        J = pin.computeFrameJacobian(robot.model,robot.data,q,IDX,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        X[:,i] = situationOT(robot.data.oMf[IDX])
        traj_dq[:,i] = dq
        traj_q[:,i] = q
        t[i] = i*dt
        dotX[:,i] = np.dot(J,dq)
        ddX[:,i] = dJv + np.dot(J,ddq)
    return X,dotX,ddX,traj_q,traj_dq,traj_ddq,t
        
#Loi polynomial

def loiPoly(robot,t,Vmax=10,debug=True):
    """ Création de la loi polynomial, à modifier il manque la vrai valeur des coeff"""
    qf = np.array([math.pi/3, +6*math.pi/4])
    a0, a1, a2, a3, tf = calcCoeff(Vmax, robot,qf) # a modifier pour le calculer seulement a la premiere itération
    #print("t \t",t)
    if(t == 0 and debug==True):
        print("a0 : \t",a0)
        print("a1 : \t",a1)
        print("a2 : \t",a2)
        print("a3 : \t",a3)
        print("tf : \t",tf)
    q = np.zeros(robot.nq)
    dq = np.zeros(robot.nv)
    ddq = np.zeros(robot.nv)
    for i in range(robot.nq):
        q[i] = a0[i] + a1[i]*t + a2[i]*(t**2) + a3[i]*(t**3) 
        dq[i] = a1[i] + 2*a2[i]*t + 3*a3[i]*t**2
        ddq[i] = 2*a2[i] + 6*a3[i]*t
        if(t>=tf[i]):
            q[i] = a0[i] + a1[i]*tf[i] + a2[i]*(tf[i]**2) + a3[i]*(tf[i]**3) #2 dimension 
            dq[i] = a1[i] + 2*a2[i]*tf[i] + 3*a3[i]*tf[i]**2
            ddq[i] = 0
    return q,dq,ddq

#Position initial du robot
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
    """retourne la loi avec série de fournier """
    q = np.array([0.1*np.cos(2*math.pi*t), 0])
    vq = np.array([-0.2*math.pi*np.sin(2*math.pi*t),0])
    aq = np.array([-0.4*math.pi**2*np.cos(2*math.pi*t),0])
    return  q,vq,aq


def simulateurVerif(N,robot):
    """ Simutalteur de vérification de la jacobienne """
    BASE = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    IDX = robot.model.getFrameId("tcp")
    dt = 1e-4
    X = np.zeros((N,3))
    t = np.zeros(N)
    dotXJac = np.zeros(X.shape)
    ddX_traj = np.zeros(X.shape)
    for i in range(N):
        q,dq,ddq = loiPoly(robot,i*dt,Vmax=4)
        robot.forwardKinematics(q) #update joint 
        pin.updateFramePlacements(robot.model,robot.data) #update frame placement
        DJ = pin.computeJointJacobiansTimeVariation(robot.model,robot.data,q,dq)
        if(i*dt == 0.5):
            print("qpoint = \t",dq)
            print("q \t",q)
            print("J \t",J)
        ddx = np.dot(DJ,dq)
        print("acceleration OT : ",ddx)
        ddX_traj[i,:] = np.array([ddx[0],ddx[2],ddx[4]])
        J = adaptJacob(pin.computeFrameJacobian(robot.model,robot.data,q,IDX,BASE))
        X[i,:] =    adaptSituation(situationOT(robot.data.oMf[IDX]),q)
        dotXJac[i,:] = np.dot(J,dq)
        t[i] = i*dt
        robot.display(q)
        time.sleep(dt)
    dotXDiff = calculDotX(X,dt)
    ddXDiff = calculDotX(dotXDiff,dt)

    print("shape dotX avec différences finies\t ",dotXDiff.shape)
    print(X[N-1,:])
    if PLOT:
        plt.plot(t,dotXJac[:,0],label="avec Jacobienne")
        plt.plot(t[:len(t)-1],dotXDiff[:,0],".",label="avec différences finis")
        plt.plot(t,X[:,0],label="position OT selon l'axe X")
        plt.title("position et dérivé situation OT selon axe x")
        plt.ylabel("m ou m/s")
        plt.xlabel("seconde")
        plt.legend()
        plt.figure()
        plt.plot(t,X[:,1],label="position selon axe z en m")
        plt.plot(t,dotXJac[:,1],label="avec Jacobienne en m/s")
        plt.plot(t[:len(t)-1],dotXDiff[:,1],".",label="avec différences finis en m/s")
        plt.title("position et dérivé situation OT selon axe z")
        plt.ylabel("m ou m/s")
        plt.xlabel("seconde")
        plt.legend()
        plt.figure()
        plt.plot(t,ddX_traj[:,0],label="accélération linéaire selon x")
        plt.plot(t,ddX_traj[:,1],label="accélération linéaire selon z")
        plt.plot(t,ddX_traj[:,2],label="accélération angulaire selon y")
        plt.plot(t[:len(t)-2],ddXDiff[:,0],".",label="x avec différences finis en m/s")
        plt.ylabel("m/s2 ou rad/s2")
        plt.xlabel("seconde")
        plt.title("acceleration de l'ot")
        plt.legend()
        plt.show()

def computeError(Xconsigne,Xactuel,dotXconsigne,dotXactuel):
    """ Renvois l'erreur de la situation de l'organe terminal et l'erreur de ça vitesse"""
    epsX = (Xconsigne-Xactuel)
    epsdotX = (dotXconsigne-dotXactuel)
    return epsX,epsdotX

def simuLoiCommande(robot):
    BASE = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    IDX = robot.model.getFrameId("tcp")
    dt = 1e-2
    N = 300
    Xc,dotXc,qc,dqc,t  = getTraj(N,robot,IDX,dt,loi='R',V=5) # on récupére la consigne 
    q = robot.q0 
    dq = np.zeros(robot.nq) #accélération des joint à l'instant initial
    traj_OT = np.zeros(Xc.shape)
    traj_dotOT = np.zeros(dotXc.shape)
    deltaX = np.zeros(Xc.shape[1])
    t = np.zeros(N)

    for i in range(N):
        robot.forwardKinematics(q) #update joint 
        pin.updateFramePlacements(robot.model,robot.data) #update frame placement
        J = adaptJacob(pin.computeFrameJacobian(robot.model,robot.data,q,IDX,BASE)) #calcul de la jacobienne
        X = adaptSituation(situationOT(robot.data.oMf[IDX]),q) #deltaX
        dotX = np.dot(J,dq)
        deltaX,deltaDotX = computeError(Xc[i,:],X,dotXc[i,:],dotX)
        q = loiCommande1(deltaX,1,J,q)
        traj_OT[i,:] = X
        traj_dotOT[i,:] = dotX
        t[i] = i*dt
        robot.display(q)
        time.sleep(dt)

    robot.forwardKinematics(q) #update joint 
    pin.updateFramePlacements(robot.model,robot.data) #update frame placement
    X= adaptSituation(situationOT(robot.data.oMf[IDX]),q)
    if PLOT: 
        plt.plot(t,traj_OT[:,0],label="position OT selon axe x")
        plt.plot(t,traj_OT[:,1],label="position OT selon axe z")
        plt.plot(t,traj_OT[:,2],label="orientation OT")
        plt.plot(t,Xc[:,0],label="position consigne selon axe x")
        plt.plot(t,Xc[:,1],label="position consigne selon axe z")
        plt.plot(t,Xc[:,2],label="orientation consigne")
        plt.legend()
        plt.savefig("asservissementPposition")
        plt.figure()
        plt.plot(t,dotXc[:,0],label="vitesse consigne axe x")
        plt.plot(t,traj_dotOT[:,0],label="vitesse OT axe x")
        plt.plot(t,dotXc[:,1],label="vitesse consigne axe y")
        plt.plot(t,traj_dotOT[:,1],label="vitesse OT axe y")
        plt.plot(t,dotXc[:,2],label="vitesse angulaire de la consigne")
        plt.plot(t,traj_dotOT[:,2],label="vitesse angulaire OT")
        plt.legend()
        plt.show()
    
#Jacobienne

def jacobienneValidation(J,dq):
    dotX = np.dot(J,dq)
    return dotX

def ComputeAngularVelocities(dq):
    """
        This function compute the angular velocities using the recursive equation
            wj = wj-1 + sigLinej*dotqjzj0 wher z0j is the axis of frame j in the frame 0
        w0 = 0
        sigLinej = 1 always beacause the joint robot are all revolutional
    """

    w = [np.zeros((3))]
    for j in range(1,robot.model.njoints):
        jointIndex = robot.model.getFrameId(robot.model.names[j])
        zj0 = robot.data.oMf[jointIndex].rotation[:,2]
        wj0 = w[j-1] + zj0*dq[j-1]
        w.append(wj0)
    #print(w)
    return np.array(w)

def getdjv(robot,q,v,a):
    """this function return the product of the derivative Jacobian times the joint velocities """ 
    IDX = robot.model.getFrameId("tcp")
    robot.forwardKinematics(q,v,0*a)
    dJv = np.hstack( (pin.getFrameClassicalAcceleration(robot.model,robot.data,IDX,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear ,pin.getFrameAcceleration(robot.model,robot.data,IDX,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).angular))
    robot.forwardKinematics(q,v,a)
    return dJv


def ComputedotJTimesdotQ(robot,dq):
    """
        this function compute the multiplication dJdq and return the current OT acceleration using a recursive algorithm derived form second kinematic model
        Appendix 10.3 form Modeling, Indetification and Control written by khalil

        aij var i in frame j 

        useless because there is a pinochio function that do it 
    """
    IDX = robot.model.getFrameId("tcp")
    psiprec0 = np.zeros(3)
    deltaprec0 = np.zeros(3)
    Uprec0 = np.zeros((3,3))
    pin.updateFramePlacements(robot.model,robot.data) #update frame placement
    w = ComputeAngularVelocities(dq)

    for i in range(1,robot.model.njoints): #start to 1 because of joint 0, is the base link
        jointIndex = robot.model.getFrameId(robot.model.names[i])#to get the frame index of joint i
        zi0 = robot.data.oMf[jointIndex].rotation[:,2]# unite vecteur zi in the frame 0, (base workd)
        psii0 = np.dot(robot.data.oMf[jointIndex].rotation,psiprec0) + np.cross(w[i],dq[i-1]*zi0)
        Ui0 = skew(psii0) + np.dot(skew(w[i]),skew(w[i]))
        deltai0 = np.dot(robot.data.oMf[jointIndex].rotation,(deltaprec0 + np.dot(Uprec0,-robot.data.oMf[jointIndex].translation)))
        Uprec0 = Ui0
        deltaprec0 = deltai0
        psiprec0 = psii0
    return np.concatenate( (deltai0,psii0))
        

def calculDotX(X,dt):
    """ Calcul de Xpoint avec les différences finis, Xpoint = DeltaX/Dt en [m/s m/s rad/s ] """
    dotX = np.zeros((X.shape[0]-1,X.shape[1]))
    for i in range(X.shape[0]-1):
        dotX[i,:] = (X[i+1,:] - X[i,:])/dt
    
    return dotX 

def loiCommande1(deltaX,Kp,J,q):
    """ calcul du la boucle ouverte """
    print("q\t",q)
    deltaQ = np.dot(pinv(J),Kp*deltaX)
    q = moveRobot(q,deltaQ)
    return q 

def loiCommande2(deltaDotX,K,J,q):
    """ Commande Seulement en vitesse """
    deltaDotQ = K*np.dot(pinv(J),deltaDotX)

    deltaQ = deltaDotQ*1e-2 
    print("deltaQ\t ",deltaQ)
    q = moveRobot(q,deltaQ)
    return q,deltaQ

def moveRobot(q,deltaQ):
    """ fonction qui donne le mouvement du robot"""
    q += deltaQ
    return q

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
    
    return X[0],X[1],Xp[1] #Xp[0] avant

def robotDynamicWithForce(robot,input,q,vq,aq,dt,IDX):
    """ 
    Dynamic of the robot calculator for postion/speed control 
    tau =  input + G

    tau_new = J't.f
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

            f = pinv(Jt)tau_new
    """

    A = pin.crba(robot.model,robot.data,q) # compute mass matrix
    H = pin.rnea(robot.model,robot.data,q,vq,np.zeros(aq.shape[0]))  # compute dynamic drift -- Coriolis, centrifugal, gravity
    tau = input
    X = np.array([q,vq])
    Xp = np.array([vq,np.dot(pinv(A),(tau-H))]) # rajout d'un fois 2 on ne sais pas trop pourquoi ( ça parraissait cohérent dans la tete de jo)
    X += Xp*dt

    q = X[0]
    vq = X[1]
    aq = Xp[1]

    robot.forwardKinematics(q) #update joint 
    pin.updateFramePlacements(robot.model,robot.data) #update frame placement

    print("force using pinnochio ",robot.data.Fcrb)
    A = pin.crba(robot.model,robot.data,q) # compute mass matrix
    #H = pin.rnea(robot.model,robot.data,q,vq,aq)  # compute dynamic drift -- Coriolis, centrifugal, gravity
    H = pin.rnea(robot.model,robot.data,q,vq,np.zeros(aq.shape[0]))  # compute dynamic drift -- Coriolis, centrifugal, gravity

    #J = adaptJacob(pin.computeFrameJacobian(robot.model,robot.data,q,IDX,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
    J =pin.computeFrameJacobian(robot.model,robot.data,q,IDX,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    tau_next = np.dot(A,aq) + H
    f= np.dot(pinv(np.transpose(J)),tau)
    return q,vq,aq,f,J,A,H
    

def adaptS(I_S):
    diagI_s = np.diag(I_S).copy()
    diagI_s[1] = 0
    diagI_s[5] = 0
    return np.diag(diagI_s)

def robot_dyn(robot,input,q,vq,aq,f,J,dt,IDX):
    """
        x vector = [ q, dq, f]
        dx vector = [dq ddq, df]
        
        dq = dq
        ddq = pinv(A),(tau-H+np.dot(np.transpose(J),f))
    """
    A = pin.crba(robot.model,robot.data,q) # compute mass matrix
    H = pin.rnea(robot.model,robot.data,q,vq,np.zeros(aq.shape[0]))  # compute dynamic drift -- Coriolis, centrifugal, gravity
    tau = input
    X = np.array([q,vq])
    Xp = np.array([vq,np.dot(pinv(A),(tau-H+np.dot(np.transpose(J),f)))]) # rajout d'un fois 2 on ne sais pas trop pourquoi ( ça parraissait cohérent dans la tete de jo)
    X += Xp*dt

    q = X[0]
    vq = X[1]
    aq = Xp[1]


def run_efort_control(robot):
    """
        this function implement the efort control scheme and run it 


    """
    N = 3000
    dt = 1e-3
    IDX = robot.model.getFrameId("tcp")
    Xdesired,dXdesired,ddXdesired,traj_q,traj_dq,traj_ddq,t = getTraj(N,robot,IDX,dt,loi='P')
    q = traj_q[:,0]
    vq = np.zeros(robot.nv)
    aq = np.zeros(robot.nv)
    trajX = np.zeros((3,N))
    trajF = np.zeros((3,N))
    robot.forwardKinematics(q) #update joints
    pin.updateFramePlacements(robot.model,robot.data) #update frame placement

    #S = computeSelectionMatrix(1,0,1,0,1,0) # position => indice 0,2,4
    #I_S = computeSelectionMatrix(0,0,0,0,0,0)#


    S = np.eye(3)   #np.array([[1,0],[0,1]]) # Sur X et Z  => X de taille 2x2
    S = np.diag([1,0,0])
    I_S = np.eye(S.shape[0]) - S  #np.eye(2)-S
    I_S = np.diag([0,1,0])
    input = pin.rnea(robot.model,robot.data,q,vq,aq)  # H compute dynamic drift -- Coriolis, centrifugal, gravity
    fd = np.array([1,1,1])
    I=0
    Id = 334#math.floor(N/3)


    print(I_S)
    for i in range(N):
        
        robot.display(q)
        X = adaptSituation(situationOT(robot.data.oMf[IDX]),q)
        trajX[:,i] = X

        #print("t :",i*dt)
        #print("X0t : ",X[0])
        q,vq,aq,f,J,A,H = robotDynamicWithForce(robot,input,q,vq,aq,dt,IDX)
        J = adaptJacob(J)
        f = adaptSituation(f,[0,0])
        #fd = f
        #print("la force exercé par le robot est égal à\t",f)
        dX = np.dot(J,vq)
        ddX = getdjv(robot,q,vq,aq)

        #print("postion error before controller",norm(adaptSituation(Xdesired[:,i])-X))

        #input,I = effort_control(adaptSituation(Xdesired[:,Id]),X,0*adaptSituation(dXdesired[:,0]),dX,0*adaptSituation(ddXdesired[:,0]),adaptSituation(ddX),fd,f,I,J,S,I_S,A,H,dt) #constant position
        input,I = effort_control(adaptSituation(Xdesired[:,i],traj_q[:,i]),X,adaptSituation(dXdesired[:,i],traj_dq[:,i]),dX,adaptSituation(ddXdesired[:,i],traj_ddq[:,i]),adaptSituation(ddX,aq),fd,f,I,J,S,I_S,A,H,dt) # tracking 
        trajF[:,i] = f
        #time.sleep(dt)

    plt.figure()
    plt.title("sur X")
    plt.plot(t,Xdesired[0,:]*np.ones(N),'r--',label="consigne")
    plt.plot(t,trajX[0,:],label="traj OT")
    plt.xlabel("temps en seconde")
    plt.ylabel("position en m")
    plt.legend()
    plt.figure()
    plt.title("sur Z")
    plt.plot(t,Xdesired[2,:]*np.ones(N),'r--',label="consigne")
    plt.plot(t,trajX[1,:],label="traj OT")
    plt.xlabel("temps en seconde")
    plt.ylabel("position en m")
    plt.figure()
    plt.title("la force en z")
    plt.plot(t,fd[1]*np.ones(N),'r--',label="consigne")
    plt.plot(t,trajF[1,:],label="force exercer par l'OT")
    plt.xlabel("temps en seconde")
    plt.ylabel("force en N")
    plt.figure()
    plt.title("la force en x")
    plt.plot(t,fd[0]*np.ones(N),'r--',label="consigne")
    plt.plot(t,trajF[0,:],label="force exercer par l'OT")
    plt.xlabel("temps en seconde")
    plt.ylabel("force en N")
    plt.legend()

    plt.show()
    


        
def run_efort_control2(robot):
    """
        this function implement the efort control scheme and run it 


    """
    N = 4000
    dt = 1e-3
    IDX = robot.model.getFrameId("tcp")
    Xdesired,dXdesired,ddXdesired,traj_q,traj_dq,traj_ddq,t = getTraj(N,robot,IDX,dt,loi='F')
    q = traj_q[:,0]
    vq = np.zeros(robot.nv)
    aq = np.zeros(robot.nv)
    trajX = np.zeros((Xdesired[:,0].shape[0],N))
    trajF = np.zeros((Xdesired[:,0].shape[0],N))
    robot.forwardKinematics(q) #update joints
    pin.updateFramePlacements(robot.model,robot.data) #update frame placement

    #S = computeSelectionMatrix(1,0,1,0,1,0) # position => indice 0,2,4
    #I_S = computeSelectionMatrix(0,0,0,0,0,0)#



    S = np.diag([1,0,1,0,1,0])
    I_S = np.diag([0,0,0,0,0,0])
   #np.array([[1,0],[0,1]]) # Sur X et Z  => X de taille 2x2
    #I_S = np.eye(S.shape[0]) - S  #np.eye(2)-S
    input = pin.rnea(robot.model,robot.data,q,vq,aq)  # compute dynamic drift -- Coriolis, centrifugal, gravity
    fd = np.array([100,0,1,0,0,0])
    I=0
    Id = 334#math.floor(N/3)


    print(I_S)
    for i in range(N):
        
        robot.display(q)
        X = situationOT(robot.data.oMf[IDX])
        trajX[:,i] = X

        #print("t :",i*dt)
        #print("X0t : ",X[0])
        q,vq,aq,f,J,A,H = robotDynamicWithForce(robot,input,q,vq,aq,dt,IDX)
        #fd = f
        #print("la force exercé par le robot est égal à\t",f)
        dX = np.dot(J,vq)
        ddX = getdjv(robot,q,vq,aq)

        #print("postion error before controller",norm(adaptSituation(Xdesired[:,i])-X))

        #input,I = effort_control(adaptSituation(Xdesired[:,Id]),X,0*adaptSituation(dXdesired[:,0]),dX,0*adaptSituation(ddXdesired[:,0]),adaptSituation(ddX),fd,f,I,J,S,I_S,A,H,dt) #constant position
        input,I = effort_control(Xdesired[:,i],X,dXdesired[:,i],dX,ddXdesired[:,i],ddX,fd,f,I,J,S,I_S,A,H,dt) # tracking 
        trajF[:,i] = f
        #time.sleep(dt)
    print("f =",f)
    plt.figure()
    plt.title("sur X")
    plt.plot(t,Xdesired[0,:]*np.ones(N),'r--',label="consigne")
    plt.plot(t,trajX[0,:],label="traj OT")
    plt.xlabel("temps en seconde")
    plt.ylabel("position en m")
    plt.legend()
    plt.figure()
    plt.title("sur Z")
    plt.plot(t,Xdesired[2,:]*np.ones(N),'r--',label="consigne")
    plt.plot(t,trajX[2,:],label="traj OT")
    plt.xlabel("temps en seconde")
    plt.ylabel("position en m")
    plt.figure()
    plt.title("la force en z")
    plt.plot(t,fd[2]*np.ones(N),'r--',label="consigne")
    plt.plot(t,trajF[2,:],label="force exercer par l'OT")
    plt.xlabel("temps en seconde")
    plt.ylabel("force en N")
    plt.figure()
    plt.title("la Moment en x")
    plt.plot(t,fd[0]*np.ones(N),'r--',label="consigne")
    plt.plot(t,trajF[0,:],label="force exercer par l'OT")
    plt.xlabel("temps en seconde")
    plt.ylabel("force en N")
    plt.legend()

    plt.show()
    





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

    Kf = 0.1
    Kdf = 0
    KIf = 0

    #if(det(J) == 0):
        #print("c'est J le probleme j'ai jurer")
    pclt = PCLT(Xd,X,dXd,dX,ddXd,ddXn,pinv(J),A,S)
    #fcl,I = FCL(fd,f,dX,np.transpose(J),I,dt,Kf,Kdf,I_S,KIf)
    fcl,I = FCL(fd,f,dX,np.transpose(J),I,dt,Kf,Kdf,I_S,KIf)

    return (pclt + fcl + H),I

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
    print("deltaF = ",out)
    return np.dot(Jt,out),I



def skew(v):
    """ 
        take the 3x1 vector and return the skew-symetric matrix
    """
    if(v.size != 3 ):
        print(" v need to be a 3x1 vector")
        return None
    else:
        return np.array([[0,-v[2],v[1]],
                        [v[2],0,-v[0]],
                        [-v[1],v[0],0]])

def simulator(robot,espace="OT"):
    """
    
    """
    N = 500
    BASE = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    IDX = robot.model.getFrameId("tcp")
    dt = 1e-2
    Xc,dotXc,qc,dqc,t = getTraj(N,robot,IDX,dt,loi='R',V=5)
    q = robot.q0    
    dq = np.zeros(robot.nv) #accélération des joint à l'instant initial
    ddq = np.zeros(robot.nv)
    traj_OT = np.zeros(Xc.shape)
    traj_dotOT = np.zeros(dotXc.shape)
    I = 0
    for i in range(N):
        
        robot.forwardKinematics(q) #update joint 
        pin.updateFramePlacements(robot.model,robot.data) #update frame placement
        J = pin.computeFrameJacobian(robot.model,robot.data,q,IDX,BASE)
        X = situationOT(robot.data.oMf[IDX])
        print(adaptPlanarRotation(robot.data.oMf[IDX].rotation))
        J = adaptJacob(J) # adadptedJ
        X = adaptSituation(X,q) #adaptedX
        dotX = np.dot(J,dq)
        if espace == "OT":
            deltaX,deltaDotX = computeError(Xc[i,:],X,dotXc[i,:],dotX) #dotXc[4,:] 
            outController,I = controllerPID(deltaX,deltaDotX,I,dt,Kp=62.5,Kd=20,Ki=0.2)
            print("output of controller \t ", outController)
            Jplus = pinv(J)
            Jt = transpose(J)
            inRobot = np.dot(Jt,outController)
        elif espace == "Joint":
            deltaQ,deltaDotQ = computeError(qc[N-1,:],q,0,dq)
            Kp,Kd,Ki = calcGain(robot,4)
            outController,I = controllerPID(deltaQ,deltaDotQ,I,dt,Kp,Kd,Ki)
            inRobot = outController

    
        q,dq,ddq = robotDynamic(robot,inRobot,q,dq,ddq,dt)
        traj_OT[i,:] = X
        traj_dotOT[i,:] = dotX
        # Display of the robot
        robot.display(q)

    
    print("position x OT" + "\tposition consigne",X[0],Xc[N-1][0])#Xc[N-1[0],:])
    print("position y OT" + "\tposition consigne",X[1],Xc[N-1][1])#Xc[N-1,:][1])
    print("orientation OT" + "\torientation consigne",X[2],Xc[N-1][2])#Xc[N-1,:][2])
    if PLOT: 
        plt.plot(t,traj_OT[:,0],label="position OT selon axe x")
        plt.plot(t,traj_OT[:,1],label="position OT selon axe y")    
        plt.plot(t,Xc[:,0],label="position consigne selon axe x")
        plt.plot(t,Xc[:,1],label="position consigne selon axe y")
        plt.legend()
        plt.figure()
        plt.plot(t,traj_dotOT[:,0],label="vitesse OT axe x")
        plt.plot(t,traj_dotOT[:,1],label="vitesse OT axe y")
        plt.legend()
        plt.show()


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

def calcGain(robot,w):
    """
        function that compute the 
    """
    A = robot.data.M
    f = robot.model.friction
    Kp = []
    Kd = []
    KI = []
    for i in range(robot.model.nq):
        Kp.append(3*A[i,i]*w**2)
        Kd.append(3*A[i,i]*w - f[i])
        KI.append(A[i,i]*w**3)

    Kp = np.diag(Kp)
    Kd = np.diag(Kd)
    KI = np.diag(KI)
    return Kp,Kd,KI
    

def PCLT(Xd,X,dXd,dX,ddXd,ddXn,Jp,A,S):
    """
        Position control law

        Computed with 
        
    """
    outcomputedTorqueController = np.transpose(computedTorqueController(Xd,X,dXd,dX,ddXd,ddXn,eye(X.shape[0]),eye(X.shape[0]),np.zeros(X.shape)))
    #print("out controller avant ",outcomputedTorqueController)
    outcomputedTorqueController = np.dot(S,outcomputedTorqueController)
    print("DeltaX = ",outcomputedTorqueController)
    AJp = np.dot(A,Jp)
    outcomputedTorqueController = np.dot(AJp,outcomputedTorqueController)
    return  outcomputedTorqueController


def computeSelectionMatrix(s1,s2,s3,s4,s5,s6):
    """ 
        Compute the selection Matrix useful for the effort control
    """

    if( (s1==1 and s4 == 1) or (s2 == 1 and s5 == 1) or (s3 == 1 and s6 == 1) ):
        print("Warning we can't control position and force along the same axis")
    return np.diag([s1,s2,s3,s4,s5,s6])
    


def carthesianPolynomialLaw(t,x0,xf,Vmax=10,debug=True):
    """
        Polynomial Law in the carthesian space
    """
    a0, a1, a2, a3, tf = calcCoeffCarthesianLaw(Vmax, robot,xf,x0) # a modifier pour le calculer seulement a la premiere itération
    if(t == 0 and debug==True):
        print("a0 : \t",a0)
        print("a1 : \t",a1)
        print("a2 : \t",a2)
        print("a3 : \t",a3)
        print("tf : \t",tf)
    X = np.zeros(6)
    Xpoint = np.zeros(6)
    Xpointpoint = np.zeros(6)
    for i in range(6):
        X[i] = a0[i] + a1[i]*t + a2[i]*(t**2) + a3[i]*(t**3) #+ a4[i]*(t**4) + a5[i]*(t**5)
        Xpoint[i] = a1[i] + 2*a2[i]*t + 3*a3[i]*t**2 #+ 4*a4[i]*t**3 + 5*a5[i]*t**4
        Xpointpoint[i] = 2*a2[i] + 6*a3[i]*t #+ 12*a4[i]*t**2 + 20*a5[i]*t**3
        if(t>=tf[i]):
            X[i] = a0[i] + a1[i]*tf[i] + a2[i]*(tf[i]**2) + a3[i]*(tf[i]**3) #+ a4[i]*(tf[i]**4) + a5[i]*(tf[i]**5)
            Xpoint[i] = a1[i] + 2*a2[i]*tf[i] + 3*a3[i]*tf[i]**2 #+ 4*a4[i]*tf[i]**3 + 5*a5[i]*tf[i]**4
            Xpointpoint[i] = 2*a2[i] + 6*a3[i]*tf[i] #+ 12*a4[i]*tf[i]**2 + 20*a5[i]*tf[i]**3  
    return X,Xpoint,Xpointpoint


def calcCoeffCarthesianLaw(Vmax, robot, xf,x0):
    """ 
        Similar than in the joint space
    """
    a0 = np.zeros(6)
    a1 = np.zeros(6)
    a2 = np.zeros(6)
    a3 = np.zeros(6)
    tf = np.zeros(6)
    DeltaX = xf - x0 #robot.data.oMf[0]
    
    for i in range(6):
        tf[i] = (3/2)*(DeltaX[i]/Vmax)
        if(tf[i] != 0):
            a0[i] = x0[i]                  # Contrainte position initiale = qinit
            a1[i] = 0                   # Containte position initiale = 0
            a2[i] = (3*DeltaX[i])/(tf[i]**2) 
            a3[i] = (-2*DeltaX[i])/(tf[i]**3)
        

    return a0,a1,a2,a3,tf

def getCarthesianTraj(robot,N,dt):
    """
        This function return the carthesian polinomial law in the carthesian space

        out is a 6xN vector 

        useless function need to be completed
    """
    X0 = situationOT(robot.data.oMf[robot.model.getFrameId("tcp")])
    Xf = np.array([1.25,0,0.6,0,0,0])
    X = np.zeros((6,N))
    dX = np.zeros((6,N))
    ddX = np.zeros((6,N))
    t = np.zeros(N)
    for i in range(N):
        X[:,i],dX[:,i],ddX[:,i] = carthesianPolynomialLaw(i*dt,X0,Xf)
        t[i] = i*dt
    
    return X,dX,ddX
    

def run(robot):
    """
        simulate the computed torque control law


    """
    dt = 1e-3
    N = 10000
    BASE = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    IDX = robot.model.getFrameId("tcp")
    
    q = robot.q0
    vq = np.zeros(robot.nv)
    aq = np.zeros(robot.nv)

    Xdesired,dXdesired,ddXdesired,traj_q,traj_dq,traj_ddq,t = getTraj(N,robot,IDX,dt,loi='R')
    q = traj_q[:,0]
    vq = np.zeros(robot.nv)
    aq = np.zeros(robot.nv)
    trajX = np.zeros(Xdesired.shape)

    robot.forwardKinematics(q) #update joints
    pin.updateFramePlacements(robot.model,robot.data) #update frame placement
    Xinit = situationOT(robot.data.oMf[robot.model.getFrameId("tcp")])
    Xf = np.array([1.15,0,0.85,0,0,0])
    dXf = np.zeros(Xf.shape)
    ddXf = np.zeros(Xf.shape)

    In = math.floor(N/4)
    for i in range(N):
        J = pin.computeFrameJacobian(robot.model,robot.data,q,IDX,BASE)
        G = pin.rnea(robot.model, robot.data, q, np.zeros(robot.nv), np.zeros(robot.nv)) #gravity matrix
        A = pin.crba(robot.model,robot.data,q) # compute mass matrix
        H = pin.rnea(robot.model,robot.data,q,vq,np.zeros(aq.shape))  # compute dynamic drift -- Coriolis, centrifugal, gravity
        robot.forwardKinematics(q) #update joint
        X = situationOT(robot.data.oMf[IDX])
        pin.updateFramePlacements(robot.model,robot.data) #update frame placement
        Jp = pinv(J)
        print("Jp ", Jp)
        dX = np.dot(J,vq)
        ddXn = getdjv(robot,q,vq,aq)
        
        trajX[:,i] = X
        tau = computedTorqueController(Xdesired[:,i],X,dXdesired[:,i],dX,ddXdesired[:,i],ddXn,Jp,A,H) #tracking
        #tau = computedTorqueController(Xdesired[:,In],X,0*dXdesired[:,In],dX,0*ddXdesired[:,In],ddXn,Jp,A,H) #constant position 

        q,vq,aq = robotDynamic(robot,tau,q,vq,aq,dt)
        robot.display(q)
        


    plt.figure()
    plt.title("suivi de trajectoire sur l'axe z, avec une position constante")
    plt.plot(t,Xdesired[2,:]*np.ones(N),'r--',linewidth= 2,label="la consigne sur z")
    plt.plot(t,trajX[2,:],label="trajectoire OT sur z")
    plt.xlabel('temps en seconde')
    plt.ylabel('position en m')
    plt.legend()
    plt.show()
        
def main(robot):
    """
        this function is to implement quick test
    """
    N = 1000
    dt = 1e-3
    Xdesired,dXdesired,ddXdesired,traj_q,traj_dq,traj_ddq,t = getTraj(N,robot,0,dt,loi='R')

    dXverif = calculDotX(Xdesired,dt)
    plt.figure()
    plt.title("verif J")
    plt.plot(t,Xdesired[0,:]*np.ones(N),'r--',linewidth= 2,label="calculer ")
    plt.plot(t,dXverif[0,:],label="diff finis")
    plt.xlabel('temps en seconde')
    plt.ylabel('position en m')
    plt.legend()
    plt.show()



    

def computedTorqueController(Xd,X,dXd,dX,ddXd,ddXn,Jp,A,H):
    """
            this is the controller of the computed torque control 

            she compute the error, and return the tau ( corresponding to U(t) )


            Kp = wj²
            Kd = 2zetawj
"""
    zeta = 1
    wj = 1 #math.sqrt(200)*math.pi 1
    kp = wj**2
    kd = 2*zeta*wj
    kp = kp*np.eye(X.shape[0])#10 on multiplie par Jp car nous somme dans l'espace de l'OT
    kd = kd*np.eye(X.shape[0])#100
    ex = Xd-X
    edx = dXd-dX
    W = np.dot(kp,ex)+np.dot(kd,edx)+ddXd-ddXn
    print("W shape :",W.shape)
    jpw = np.dot(Jp,W)

    tau = np.dot(A,jpw) + H
    return tau

    
Main = True
package_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/Modeles/'
urdf_path = package_path + 'planar_2DOF/URDF/planar_2DOF_TCP.urdf'

robot = RobotWrapper()
if __name__ == '__main__':
    robot = RobotWrapper.BuildFromURDF(urdf_path,package_path,verbose=True)
    robot.initViewer(loadModel=True)
    robot.display(robot.q0)
    #run(robot)
    #main(robot)
    #simulateurVerif(1000,robot)
    run(robot)
>>>>>>> main
