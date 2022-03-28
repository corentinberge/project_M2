
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




# chargement du model 
def orientationVector(R):
    """ calcul du vecteur d'orientation, selon les angles de Roulis Tangage Lacet, la fonction prend en entrée la matrice de rotation
    et renvois rotation autour de  x, y et z """
    t31 = R[2,0]
    if(abs(t31) == 1 ): #Beta = +/- 1 
        t12 = R[0,1]
        t22 = R[1,1]
        beta = -t31*math.pi/2 # gamma et alpha indeterminé
        gamma = math.pi/6 # choix arbitraire 
        alpha = (t12/t22) + gamma
    else:
        beta = math.asin(-t31)
        t21 = R[1,0]
        t11 = R[0,0]
        t32 = R[2,1]
        t33 = R[2,2]
        alpha = math.atan2(t21,t11)
        gamma = math.atan2(t32,t33)
    return [gamma,beta,alpha]

def vecX(X):
    """prend en entrée un SE3 et resort le vecteur X représentant la position de ce SE3 sous forme [translation rotation] """
    Rot = orientationVector(X.rotation)
    P = X.translation
    return np.concatenate((P,Rot),axis=None)

def forceControlLaw(f,dX,fd,S,I_f,Kf = 0 ,Kfi = 0,Kfd = 0,dt = 1e-2): #faux a revoir (nvx de l'intégral)
    """calcul de la loi de control FCL, resort la force extérieur
        f force exercé par le robot J'*torqe,  dX vitesse actuel de l'organe terminal fd force désiré, Kf gain proportionnel de la force, Kfi prop force de l'action I
        Kfd gain prop action D, dt est le quantum de temps sert à l'action intégral (j'ai un doute pour l'effet integral verifier calcul !!), i index du simulateur 
        """
    Df = fd-f
    I_f += Df*dt
    out = np.dot((np.eye(S.shape[0])-S),Kfi*dt*Df + Kf*Df + fd)
    return out,I_f

def positionControlLaw(X,dX,Xd,dXd,ddXd,J,dJ,dq,S,Kd = 0,Kp = 0): 
   """ Calcul de la loi de position, ressort ddq 
   X position de la l'OT, dX vitesse organe terminal (J*dq), Xd position organe terminal désiré, dXd vitesse organe terminal désiré 
   ddXd accélération organe terminal désiré, Kd gain dX, Kp gain X""" 
   Dx = Xd-X
   DdX = dXd - dX
   out = np.dot(np.dot(pinv(J),S),ddXd + Kd*DdX + Kp*Dx - np.dot(dJ,dq))
   return out

def positionControlLaw2(tool,goal,ddXd,J,dJ,dq,S,Kd = 0,Kp = 0): 
   """ Calcul de la loi de position, ressort ddq 
   X position de la l'OT, dX vitesse organe terminal (dJ*dq), Xd position organe terminal désiré, dXd vitesse organe terminal désiré 
   ddXd accélération organe terminal désiré, Kd gain dX, Kp gain X""" 
   Dx = vecX(tool.inverse()*goal)
   DdX = pin.log(tool.inverse()*goal).vector
   out = np.dot(np.dot(pinv(J),S),ddXd + Kd*DdX + Kp*Dx - np.dot(dJ,dq))
   return out

def simulateur(nIter,robot,goal,ID,S,qInit):
    """ Simulation du robot
    nIter nombre d'itération de la boucle temps total = nSample*dt, robot est un objet RobotWrapper, goal est la position de l'organe terminal souhaité 
    ID index de la frame de l'OT, S matrix de selection, qInit, config Initial du robot"""

    model,data = robot.model,robot.data
    NQ = robot.nq
    NV = robot.nv
    LOCAL,WORLD = pin.ReferenceFrame.LOCAL,pin.ReferenceFrame.WORLD
    q = qInit
    dq = np.zeros(NV) # à trouver une valeur cohérente 0?
    print(dq)
    Xd = vecX(goal)
    
    ddXd = np.zeros(6)#idem accélération
    fd = np.zeros(6)
    dt = 1e-2
    fe = np.zeros(6) #force que le robot applique #calculer avec J^t*torque
    I_f = 0
    for i in range(nIter-1):
        robot.forwardKinematics(q) #update joint 
        pin.updateFramePlacements(robot.model,robot.data) #update frame placement  
        Mtool = robot.data.oMf[ID] #position organe terminal 
        dXd = pin.log(Mtool.inverse()*Mgoal).vector
        X = vecX(Mtool)

        A = pin.crba(robot.model,robot.data,q) # calcul de la matrice de masse
        C = pin.computeCoriolisMatrix(robot.model,robot.data,q,dq) # calcul de la matrice de coriolis
        G = pin.computeGeneralizedGravity(robot.model,robot.data,q) # calcul du vecteur de gravité 
        b = pin.rnea(robot.model,robot.data,q,dq,np.zeros(NV))
        J = pin.computeFrameJacobian(robot.model,robot.data,q,ID,LOCAL) # calcul de la jacobienne 

        dX = np.dot(J,dq)
        ddq = positionControlLaw2(Mtool,goal,ddXd,J,robot.data.dJ,dq,S,Kd=1,Kp=1)
        #print("accélération calculer avec la PCL \t ",ddq)
        f, I_f = forceControlLaw(fe,dX,fd,S,I_f,Kf = 1,Kfi = 1,Kfd = 1,dt = 1e-2) #force externe de contact

        #print("force calculer avec FCL \t ",f)
        #torque = pin.rnea(robot.model,robot.data,q,dq,ddq)
        torque = np.dot(A,ddq) + np.dot(C,dq) + G + np.dot(np.transpose(J),f) #modèle dynamique

        fe = np.dot(pinv(np.transpose(J)),torque)

        #ddq = np.dot(inv(A),(torque-(np.dot(C,dq) + G + np.dot(np.transpose(J),fe))))
        ddq = np.dot(inv(A),(torque-b-np.dot(np.transpose(J),f)))
        dq += ddq*dt
        q = pin.integrate(robot.model,q,dq*dt)

        if not i % 3: # Only display once in a while ... 
            robot.display(q)
            
        time.sleep(1e-2)
    print("différence de position \t", Xd-X)

def place(robot,name,M):
    robot.viewer.gui.applyConfiguration(name,se3ToXYZQUAT(M))
    robot.viewer.gui.refresh()

def Rquat(x,y,z,w): 
    q = pin.Quaternion(x,y,z,w)
    q.normalize()
    return q.matrix()

def initFrame(Mgoal,robot,gv):
    IDX_tool = robot.model.getFrameId("tcp")
    IDX_BASE = robot.model.getFrameId("base_link")
    Mtool = robot.data.oMf[IDX_tool]
    MBase = robot.data.oMf[IDX_BASE]
    gv.addXYZaxis('world/framegoal',[1.,0.,0.,1.],.015,.4) # framecolor, width, length 
    gv.addXYZaxis('world/tool',[1.,0.,0.,1.],.015,.4) # framecolor, width, length 
    gv.addXYZaxis('world/base',[1.,0.,0.,1.],.015,.4)
    place(robot,'world/framegoal',Mgoal)
    place(robot,'world/tool',Mtool)
    place(robot,"world/base",MBase)


Main = True
path = '/home/jo/'
urdf = '/home/jo/robots/planar_2DOF/urdf/planar_2DOF_TCP.urdf'

robot = RobotWrapper.BuildFromURDF(urdf,path,verbose=True)
robot.initViewer(loadModel=True)
robot.display(robot.q0)
model = robot.model
data = robot.data

print(model)
print(data)
LOCAL,WORLD = pin.ReferenceFrame.LOCAL,pin.ReferenceFrame.WORLD
NQ = robot.nq
NV = robot.nv
q0 = robot.q0 #config intial

ddq = np.zeros(NV)
dq = rand(NV)
dt = 1e-2

q = robot.q0.copy()
F = 0

sample = 1000
traj_q = np.zeros((NQ,sample))
traj_dq = np.zeros((NV,sample))
traj_ddq = np.zeros((NV,sample))
traj_torque = np.zeros((NV,sample))
traj_torque_validation = np.zeros((NV,sample))

ID  = robot.model.getFrameId("tcp")
S = np.diagflat([1,1,1,1,1,1]) # Selection Matrix a voir la vrai valeur
gv = robot.viewer.gui


if Main == True:
    q = rand(NQ) #définition placement initial
    Mgoal = pin.SE3(Rquat(0.,0.,0,0),np.matrix([1.25,0,1]).T) #definition de la target
    initFrame(Mgoal,robot,gv)
    simulateur(5000,robot,Mgoal,ID,S,q)
    print("Finis")

if Main == False : 
    for i in range(sample):
        M = pin.crba(robot.model,robot.data,q) # calcul de la matrice de masse
        C = pin.computeCoriolisMatrix(robot.model,robot.data,q,dq) # calcul de la matrice de coriolis
        G = pin.computeGeneralizedGravity(robot.model,robot.data,q) # calcul du vecteur de gravité 
        J = pin.computeFrameJacobian(robot.model,robot.data,q,ID,LOCAL)
        robot.forwardKinematics(q) #update joint 
        pin.updateFramePlacements(robot.model,robot.data) #update frame placement  
        X_tcp = robot.data.oMf[ID] #position organe terminal 
    
        torque = np.dot(M,ddq) + np.dot(C,dq) + G # np.transpose(J)*F #modèle dynamique
        f = np.dot(pinv(np.transpose(J)),torque)

        traj_q[:,i] = q
        traj_dq[:,i] = dq
        traj_ddq[:,i] = ddq
        traj_torque[:,i] = torque
        traj_torque_validation[:,i] = pin.rnea(robot.model,robot.data,q,dq,zero(robot.model.nv))

        alpha = torque - np.dot(C,dq) - G
        ddq = np.dot(inv(M),alpha)
        dq += ddq*dt #méthode des rectangle gauche
        q = pin.integrate(robot.model,q,dq*dt) #q+1 = integral(q-1v*dt)
    
    
        if not i % 3: # Only display once in a while ... 
            robot.display(q)
            time.sleep(1e-4)


    print("Matrice de Selection \t",S)
    #print("M : ",M)
    #print("C : ",C)
    #print("G : ",G)
    #print("torque : ",torque)
    #print("J ",J)
    print("X-tcp\t",X_tcp)

    print("X-tcp forme commande\t",vecX(X_tcp))
    print("produit matriciel X*S")
    print(np.dot(vecX(X_tcp),S))
    plt.figure()
    plt.title("torque")
    plt.plot(traj_torque[0,:],'.')
    plt.plot(traj_torque[1,:,],'.')
    plt.plot(traj_torque_validation[0,:])
    plt.plot(traj_torque_validation[1,:])

    plt.figure()
    plt.plot(traj_ddq[0,:])
    plt.plot(traj_ddq[1,:])
    #plt.show()


