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



def loiPendule(robot,t):
    """retourne la loi avec série de fournier """
    q = np.array([0,0.1*np.cos(math.pi*t),0.1*np.cos(math.pi*t),0,0,0]) + np.array([0.34,0.34,0.34,0.34,0.34,0.34])
    vq = np.array([0,-0.1*math.pi*np.sin(math.pi*t),-0.1*math.pi*np.sin(math.pi*t),0,0,0])
    aq = np.array([0,-0.1*math.pi**2*np.cos(math.pi*t),-0.1*math.pi**2*np.cos(math.pi*t),0,0,0])
    return  q,vq,aq

    

package_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + '/src/hc10_ros'
urdf_path = package_path + '/urdf/hc10.urdf'

def skew(v):
    """
        transform a vector of 3 dim in a skew matrix
    """
    sk =  np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1], v[0], 0]])
    print(sk)
    return sk

class command():
    """
        The command object 
        for 2 DOF of yaskawa, joint 2 and joint 3 sur le plan (0,X,Z)
    """
    def __init__(self):
        self.mode = 'cst' # cst for a constant position and orientation, folow to folow a trajectory 
        self.robot = RobotWrapper.BuildFromURDF(urdf_path,package_path,verbose=True)
        self.robot.initViewer(loadModel=True)
        self.robot.display(self.robot.q0)
        self.robot.viewer.gui.addXYZaxis("world/base",[1,1,1,1],.1,1) # x:rouge, y:vert, z:bleu
        self.robot.viewer.gui.addSphere("world/target",0.05,[1.0,0.0,0.0,1.0])
        self.robot.viewer.gui.addSphere("world/current",0.05, [0.0,0.9,0.0,1.0])
        self.EF_index = self.robot.model.getFrameId("tool0") # A CHANGER pour l'OT
        self.N = 50000
        self.dt = 1e-3

        if self.mode == 'suivi':
            self.trajXc,self.trajdXc,self.trajddXc,traj_q,traj_dq,traj_ddq  = self.getTraj(self.N)
            self.qinit = traj_q[:,0]
            pin.framesForwardKinematics(self.robot.model,self.robot.data,self.qinit)
            self.Xinit = self.robot.data.oMf[self.EF_index].copy()
        if self.mode == 'cst':
            self.qinit = np.array([0.34,np.pi/3,np.pi/3,0,0.34,np.pi/4])
            self.qf = np.array([np.pi/2,-0.5,0.2,0,0.5,0]) + self.qinit # move from 10deg from qinit

            # Xinit
            pin.framesForwardKinematics(self.robot.model,self.robot.data,self.qinit)
            self.Xinit = self.robot.data.oMf[self.EF_index].copy()

            # Xf 
            pin.framesForwardKinematics(self.robot.model,self.robot.data,self.qf)
            self.Xf = self.robot.data.oMf[self.EF_index].copy()
            self.robot.viewer.gui.applyConfiguration("world/target", pin.se3ToXYZQUATtuple(self.Xf))
            self.robot.viewer.gui.applyConfiguration("world/current", pin.se3ToXYZQUATtuple(self.Xinit))

            
        


        self.q = self.qinit #   reshapeVectorQTo2DOF(self.qinit)
        self.vq = np.zeros(self.qinit.shape)
        self.aq = np.zeros(self.qinit.shape)


        self.dXc = np.zeros(6)
        self.ddXc = np.zeros(6)


        #matrix initalisation
        self.A = pin.crba(self.robot.model,self.robot.data,self.qinit) # compute mass matrix
        self.H = pin.rnea(self.robot.model,self.robot.data,self.qinit,np.zeros(6),np.zeros(self.qinit.shape))  # compute dynamic drift -- Coriolis, centrifugal, gravity
        self.J = pin.computeFrameJacobian(self.robot.model,self.robot.data,self.q,self.EF_index,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        self.tau = self.H
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
        traj_q = np.zeros((6,N))
        traj_dq = np.zeros(traj_q.shape)
        traj_ddq = np.zeros(traj_q.shape) 
        t = np.zeros(N) 
        for i in range(N):
            q,dq,ddq = loiPendule(self.robot,i*dt)
            self.robot.forwardKinematics(q,dq,0*ddq)
            q2dof = q
            vq2dof = dq
            aq2dof = ddq
            djv = self.getdjv()
            pin.updateFramePlacements(self.robot.model,self.robot.data) #update frame placement 
            J = pin.computeFrameJacobian(self.robot.model,self.robot.data,q,self.EF_index,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            X.append(self.robot.data.oMf[self.EF_index].copy())    
            traj_q [:,i] = q
            traj_dq[:,i] = dq
            traj_ddq[:,i] =ddq
            t[i] = i*dt
            dotX.append(np.dot(J,dq) + djv )
            ddX.append(  djv + np.dot(J,ddq)) 
        return X,np.array(dotX),np.array(ddX),traj_q,traj_dq,traj_ddq 




    def orientationEuler(self,R):
        """ Renvois l'orientation selon la valeurs des angles d'euler  
        prend une matrice de rotation 3x3 en entrée
        
        a changer mettre les quaternions ici ou roll pitch yaw 

        """
        if(abs(R[2,2]) != 1):
            psi = math.atan2(R[0,2],-R[1,2])
            theta = math.acos(R[2,2])
            phi = math.atan2(R[2,0],R[2,1])
        else : # attention psi et phi ne sont pas définis ici phi = 2*psi => évite la division par 0 
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
        return np.array(p[0],p[2])
    
    def _measured_joint_callback(self):
        """
            each time a joint as been published on the topic then we compute all data that we need to do the control law
            q,vq,aq vector 6x1
            q2dof,vq2dof,aq2dof vector 2x1
        """



        self.q, self.vq, self.aq  = self.robotDynamic(self.tau) # 6x1
        pin.framesForwardKinematics(self.robot.model,self.robot.data,self.q)

                
        self.A = pin.crba(self.robot.model,self.robot.data,self.q) # compute mass matrix
        self.H = pin.rnea(self.robot.model,self.robot.data,self.q,self.vq,np.zeros(self.q.shape))  # compute dynamic drift -- Coriolis, centrifugal, gravity
        J6 = pin.computeFrameJacobian(self.robot.model,self.robot.data,self.q,self.EF_index,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        self.J = J6

        if self.mode == 'folow':
            Xc = self.trajXc[self.i]
        if self.mode == 'cst':
            Xc = self.Xf
        
        self.robot.viewer.gui.applyConfiguration("world/target", pin.se3ToXYZQUATtuple(Xc))

        self.Xmeasure = self.robot.data.oMf[self.EF_index].copy()
        self.robot.viewer.gui.applyConfiguration("world/current", pin.se3ToXYZQUATtuple(self.Xmeasure))
        self.dXmeasure = np.dot(self.J,self.vq)
        self.ddXmeasure = self.getdjv() # compute Jdot.qdot
        self.tau = self.computedTorqueController(Xc,self.Xmeasure,self.dXc,self.dXmeasure,self.ddXc,self.ddXmeasure) 
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
        kp= 0.5#0.2
        kd = 2*math.sqrt(kp)
        ex = self.error_EF(Xc,Xm)

        print("error in norm :",np.linalg.norm(ex))
        print("position error in norm :",np.linalg.norm(ex[0:3]))
        print("orientationnal error in norm :",np.linalg.norm(ex[3:]))

        edx = dXc-dXm 
        self.error.append(np.linalg.norm(ex))
        Jp = np.linalg.pinv(self.J) 
        W = np.dot(Jp,kp*ex + kd*edx+ddXc-ddXm) # vector 2x1  
        return np.dot(self.A,W) + self.H 
#       self.dXc = data.EFVelocity
#      self.ddXc = data.EFAcceleration

    def _publish_JointTorque(self):
        """
            put the control law here 
        """

        
    def robotDynamic(self,tau):
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
        print("tau robot dynamics :",self.tau)
        X = np.array([self.q,self.vq])
        Xp = np.array([self.vq,np.dot(np.linalg.pinv(self.A),(self.tau-self.H))])
        X += Xp*self.dt

        return X[0],X[1],Xp[1]


    def error_EF2(self,target,current):
        """
            this function compute the orientation error between target and current end effector position
            Method 4 – From Closed-loop manipulater control using quaternion feedback (Yuan, 1988) and Operational space control: 
            A theoretical and empirical comparison (Nakanishi et al, 2008)
        """
        quat_target = pin.SE3ToXYZQUAT(target)
        quat_current = pin.SE3ToXYZQUAT(current)
        n = quat_current[3]
        eps = quat_current[4:]
        nd = quat_target[3]
        epsd = quat_target[4:]


        rot_error = nd*epsd-n*eps+np.dot(skew(epsd),eps)
        p_error = quat_target[0:3]-quat_current[0:3]
        return np.hstack((p_error,rot_error))
    def error_EF(self,target,current):
        """
            this function compute the orientation error between target and current end effector position Angle/axis feedback 
            from Resolved-acceleration control of robot manipulators: A critical review with experiments (Caccavale et al, 1998)
        """
        quat_target = pin.SE3ToXYZQUAT(target)
        quat_current = pin.SE3ToXYZQUAT(current)
        R_de = np.dot(target.rotation,current.rotation.T)
        q_ed = pin.SE3ToXYZQUAT(pin.SE3(R_de,np.zeros(3)))
        n = q_ed[3]
        q_ed = q_ed[4:]/(np.linalg.norm(q_ed[4:]))


        rot_error = 2*n*q_ed
        rot_error = -rot_error
        p_error = quat_target[0:3]-quat_current[0:3]
        return np.hstack((p_error,rot_error))
        
    def getdjv(self):
        """
            this function return the product of the derivative Jacobian times the joint velocities 
        """ 
        djV = pin.getFrameClassicalAcceleration(self.robot.model,self.robot.data,self.EF_index,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return np.hstack( (djV.linear, djV.angular) )

    def run(self):
        
        self.q = self.qinit.copy()
        for self.i in range(self.N):
            self._measured_joint_callback()
            self.robot.display(self.q)
            #time.sleep(self.dt)
            self.t.append(self.i*self.dt)
            print("det J ",np.linalg.det(self.J))
            if(abs(np.linalg.det(self.J))<1e-6):
                print("singularité")
                print("valeur de J ",self.J)
                break
        t = np.array(self.t)
        e = np.array(self.error)    
        plt.figure()
        plt.title("norm error situation EF")
        plt.plot(t,e)
        plt.show()
        

        

    



if __name__ == "__main__": 
    #rospy.init_node('computed_torque_control_node')
    computed_torque = command()
    computed_torque.run()
