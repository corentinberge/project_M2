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
