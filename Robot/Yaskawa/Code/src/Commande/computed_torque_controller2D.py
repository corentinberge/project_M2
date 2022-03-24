#!/usr/bin/python
# -*- encondin: utf-8 -*
""" 
    This file contain the command_node
    2 suscriber : 
        1 for the trajectory node
        1 for the Simulation node (the robot)
    1 publisher :
        1 for the simulation node the robot
"""

#system
import os
import math
import numpy as np
import time
import matplotlib.pyplot as plt

# pinocchio 
import pinocchio as pin
from pinocchio.utils import *
from pinocchio.robot_wrapper import RobotWrapper

package_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + '/src/hc10_ros'
urdf_path = package_path + '/urdf/hc10.urdf'

def skew(v):
    """
        transform a vector of 3 dim in a skew matrix
    """
    sk =  np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1], v[0], 0]])
    print(sk)
    return sk
def reshapeJacobian(J):
    """
        this function reshape the jacobian for x and by moving joint 2 and joint 3
    """
    newJ = np.vstack((J[0,1:3],J[2,1:3]))
    #print("nouvelle jacobienne",newJ)
    return newJ


def reshapeVectorQTo2DOF(V):
    """
        This function reshape q vector to 2dof
    """
    newV = V[1:3]
    #print("nouveau vecteur",newV.shape)
    return newV

def reshape2DOFToVectorQ(V,q0):
    """
        This function reshape q vector to 2dof
        in : 
            V vecteur 2x1
            q0 vecteur 6x1
    """
    newQ = q0
    newQ[1:3]=V
   # print("nouveau vecteur",newQ.shape)
    return newQ

def loiPendule(robot,t):
        """retourne la loi avec série de fournier """
        q = np.array([0,1*np.cos(2*math.pi*t),0.1*np.cos(math.pi*t),0,0,0]) + np.array([0.34,0.34,0.34,0.34,0.34,0.34])
        vq = np.array([0,-0.1*math.pi*np.sin(math.pi*t),-0.1*math.pi*np.sin(math.pi*t),0,0,0])
        aq = np.array([0,-0.1*math.pi**2*np.cos(math.pi*t),-0.1*math.pi**2*np.cos(math.pi*t),0,0,0])
        return  q,vq,aq

class command():
    """
        The command object 
        for 2 DOF of yaskawa, joint 2 and joint 3 sur le plan (0,X,Z)
    """
    def __init__(self):
        print("package path :",package_path)
        print("computed torque construction")
        
        self.robot = RobotWrapper.BuildFromURDF(urdf_path,package_path,verbose=True)
        self.robot.initViewer(loadModel=True)
        self.robot.viewer.gui.refresh()
        self.robot.display(self.robot.q0)
        self.robot.viewer.gui.addXYZaxis("world/base",[1,1,1,1],.1,0.5) # x:rouge, y:vert, z:bleu
        self.robot.viewer.gui.addSphere("world/target",0.05,[1.0,0.0,0.0,1.0])
        self.robot.viewer.gui.addSphere("world/current",0.05, [0.0,0.0,1.0,1.0])

        self.EF_index = self.robot.model.getFrameId("tool0") # A CHANGER
        self.dt = 1e-3
        #self.dt = 0.1
        self.N = 50000

        # determine Xinit 
        self.qinit = np.array([0.0,np.pi/2,-np.pi/2,0,0,0])
        pin.framesForwardKinematics(self.robot.model,self.robot.data,self.qinit)
        self.Xinit = self.situationOT(self.robot.data.oMf[self.EF_index].copy())

        # determine traj_Xc
        self.trajXc,self.trajdXc,self.trajddXc,traj_q2dof,traj_dq2dof,traj_ddq2dof  = self.getTraj(self.N)


        #self.robot.viewer.gui.applyConfiguration("world/target", pin.se3ToXYZQUATtuple(self.Xf))

        self.q = reshapeVectorQTo2DOF(self.qinit)
        self.vq = np.zeros(self.qinit.shape)
        self.aq = np.zeros(self.qinit.shape)

        #matrix initalisation
        self.A = pin.crba(self.robot.model,self.robot.data,self.qinit) # compute mass matrix
        self.H = pin.rnea(self.robot.model,self.robot.data,self.qinit,np.zeros(6),np.zeros(self.qinit.shape))  # compute dynamic drift -- Coriolis, centrifugal, gravity
        self.tau = self.H
        self.robot.display(self.qinit)
        #print("situation EF :",self.Xinit)
        #print("couple initial demandé",self.tau)



        self.error = []
        self.t = []


    def getTraj(self,N):
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
        X = []
        dotX = []
        ddX = []
        dt = self.dt
        traj_q2dof = np.zeros((2,N))
        traj_dq2dof = np.zeros(traj_q2dof.shape)
        traj_ddq2dof = np.zeros(traj_q2dof.shape) 
        t = np.zeros(N) 
        for i in range(N):
            q,dq,ddq = loiPendule(self.robot,i*dt)
            self.robot.forwardKinematics(q,dq,0*ddq)
            q2dof = reshapeVectorQTo2DOF(q)
            vq2dof = reshapeVectorQTo2DOF(dq)
            aq2dof = reshapeVectorQTo2DOF(ddq)
            djv = self.getdjv()
            pin.updateFramePlacements(self.robot.model,self.robot.data) #update frame placement 
            #robot.display(q)
            J = reshapeJacobian(pin.computeFrameJacobian(self.robot.model,self.robot.data,q,self.EF_index,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
            X.append(self.robot.data.oMf[self.EF_index])    
            traj_dq2dof[:,i] = vq2dof
            traj_q2dof[:,i] = aq2dof
            t[i] = i*dt
            dotX.append(np.dot(J,vq2dof)+djv)
            ddX.append(  djv + np.dot(J,aq2dof)) 
        return X,np.array(dotX),np.array(ddX),traj_q2dof,traj_dq2dof,traj_ddq2dof 


    def orientationEuler(self,R):
        """ Renvois l'orientation selon la valeurs des angles d'euler  
        prend une matrice de rotation 3x3 en entrée
        
        a changer mettre les quaternions ici ou roll pitch yaw 

        """
        #print("R22 = ",R[2,2])
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
        return np.array([psi%(2*math.pi),theta%(2*math.pi),phi%(2*math.pi)])

    def orientationRTL(self,R):
        if(R[2,0] != 1):
            alpha = math.atan2(R[1,0],R[0,0])
            beta = math.atan2(-R[2,0],math.sqrt(R[0,0]**2+R[1,0]**2))
            gamma  = math.atan2(R[2,1],R[2,2]) 
        else: #attention solution dégénéré, on pose gamma = 2*alpha
            beta = -np.sign(R[2,0])*math.pi/2
            alpha = math.atan2(R[2,1],R[2,0])/(1-2*np.sign(beta))
            gamma = 2*alpha
        
        return np.array([alpha%(2*math.pi),beta%(2*math.pi),gamma%(2*math.pi)])

            

    def situationOT(self,M):
        """ 
            cette fonction permets à partir d'un objet SE3, d'obtenir un vecteur X contenant la transaltion et la rotation de L'OT (situation de l'OT)
            avec les angles d'euler classique, M est l'objet SE3, out = [ex ey ez psi theta phi]
            degenerated this function return px and py
         """
        p = M.translation
        #delta = self.orientationRTL(M.rotation) #à decommenter a terme 
        #return np.concatenate((p,delta),axis=0)
        print(p)
        return np.array([p[0],p[2]])
    
    def _measured_joint_callback(self):
        """
            each time a joint as been published on the topic then we compute all data that we need to do the control law
            q,vq,aq vector 6x1
            q2dof,vq2dof,aq2dof vector 2x1
        """
        self.q, self.vq, self.aq  = self.robotDynamic() # 6x1
        pin.framesForwardKinematics(self.robot.model,self.robot.data,self.q)
        
        
        #print("Jacobian Computed",self.J)
        self.A = pin.crba(self.robot.model,self.robot.data,self.q) # compute mass matrix
        self.H = pin.rnea(self.robot.model,self.robot.data,self.q,self.vq,np.zeros(self.q.shape))  # compute dynamic drift -- Coriolis, centrifugal, gravity
        #print("H : ",self.H)
        J6 = pin.computeFrameJacobian(self.robot.model,self.robot.data,self.q,self.EF_index,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        self.J = reshapeJacobian(J6)
        #print("jacobienne 6x6 : \n",J6)
        #print("jacobienne 2x2 : \n",self.J)


        q2dof = reshapeVectorQTo2DOF(self.q)
        vq2dof = reshapeVectorQTo2DOF(self.vq)
        aq2dof = reshapeVectorQTo2DOF(self.aq)
        

        #unpack SE3 value

        self.Xc = self.trajXc[self.i]
        self.robot.viewer.gui.applyConfiguration("world/target", pin.se3ToXYZQUATtuple(self.Xc))
        
        self.Xmeasure = self.robot.data.oMf[self.EF_index].copy()
        self.robot.viewer.gui.applyConfiguration("world/current", pin.se3ToXYZQUATtuple(self.Xmeasure))
        
        
        self.Xmeasure = self.situationOT(self.Xmeasure)

        self.dXmeasure = np.dot(self.J,vq2dof)
        self.ddXmeasure = self.getdjv() # compute Jdot.qdot
        print("shape trajDX",self.trajdXc.shape)

        self.tau = self.computedTorqueController(self.situationOT(self.Xc),self.Xmeasure,self.trajdXc[self.i,:],self.dXmeasure,self.trajddXc[self.i,:],self.ddXmeasure) 
        self._publish_JointTorque() 
        print(" -------------------------------------- ")
        
    
    def computedTorqueController(self,Xc,Xm,dXc,dXm,ddXc,ddXm): 
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
        kp= 10
        kd = 2*math.sqrt(kp)

        ex = Xc-Xm # vector 2x1
        edx = dXc-dXm # vector 2x1
        print("error ", ex)
        self.error.append(np.linalg.norm(ex))
        Jp = np.linalg.pinv(self.J) # matrix 2x2
        W = np.dot(Jp,kp*ex + kd*edx+ddXc-ddXm) # vector 2x1
        nW = reshape2DOFToVectorQ(W,self.q.copy())
        return np.dot(self.A,nW) + self.H 

#   def _trajectory_callback(self,data):
#       """ 
#          trajectory node, send a message every x seconde
#       """
#       self.Xc = data.EFPosition
#       self.dXc = data.EFVelocity
#      self.ddXc = data.EFAcceleration

    def _publish_JointTorque(self):
        """
            put the control law here 
        """

        
    def robotDynamic(self):
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

        X = np.array([self.q,self.vq])
        Xp = np.array([self.vq,np.dot(np.linalg.pinv(self.A),(self.tau-self.H))])
        X += Xp*self.dt

        return X[0],X[1],Xp[1]

    def error_EF(self,target,current):
        """
            this function compute the orientation error between target and current end effector position
        """
        quat_target = pin.SE3ToXYZQUAT(target)
        quat_current = pin.SE3ToXYZQUAT(current)
        R_de = np.dot(target.rotation,current.rotation.T)
        q_ed = pin.SE3ToXYZQUAT(pin.SE3(R_de,np.zeros(3)))
        print("q_de : ",q_ed)
        n = q_ed[3]
        q_ed = q_ed[4:]/(np.linalg.norm(q_ed[4:]))


        rot_error = 2*n*q_ed
        p_error = quat_target[0:3]-quat_current[0:3]
        return np.hstack((p_error,rot_error))
        
    def getdjv(self):
        """
            this function return the product of the derivative Jacobian times the joint velocities 
        """ 
        djV = pin.getFrameClassicalAcceleration(self.robot.model,self.robot.data,self.EF_index,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return np.array([djV.linear[0],djV.linear[2]])

    def run(self):
        
        self.q = self.qinit.copy()
        #self.Xc,self.dXc,self.ddXc,self.q_traj, self.dq_traj, self.ddq_traj = getTraj(self.N,self.robot,self.dt)

        for self.i in range(self.N):
            self._measured_joint_callback()
            self.robot.display(self.q)
            time.sleep(self.dt)
            self.t.append(self.i*self.dt)
            #print("det J ",np.linalg.det(self.J))
            if(abs(np.linalg.det(self.J))<1e-6):
                print("singularité")
                break
            print(" joint configuration ",self.q)
            print(" situation EF ", self.Xmeasure)
        t = np.array(self.t)
        e = np.array(self.error)    
        print("config error :",self.q-self.qinit)
        plt.figure()
        plt.title("norm error situation EF")
        plt.plot(t,e)
        plt.show()
        

        

    



if __name__ == "__main__": 
    #rospy.init_node('computed_torque_control_node')
    computed_torque = command()
    computed_torque.run()
