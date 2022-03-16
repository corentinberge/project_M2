""" 
    This file contain the command_node
    2 suscriber : 
        1 for the trajectory node
        1 for the Simulation node (the robot)
    1 publisher :
        1 for the simulation node the robot
"""

# system
import os
import math
import numpy as np

# pinocchio
import pinocchio as pin
from pinocchio.utils import *
from pinocchio.robot_wrapper import RobotWrapper

# ros
import rospy

# ROS MSG
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import ApplyJointEffort
from std_msgs.msg import Float64

package_path = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))) + '/src/hc10_ros'
urdf_path = package_path + '/urdf/hc10.urdf'


def reshapeJacobian(J):
    """
        this function reshape the jacobian for x and by moving joint 2 and joint 3
    """
    newJ = np.vstack((J[0, 1:3], J[2, 1:3]))
    #print("nouvelle jacobienne",newJ)
    return newJ


def reshapeVectorQTo2DOF(V):
    """
        This function reshape q vector to 2dof
    """
    newV = V[1:3]
    #print("nouveau vecteur",newV.shape)
    return newV


def reshape2DOFToVectorQ(V, q0):
    """
        This function reshape q vector to 2dof
        in : 
            V vecteur 2x1
            q0 vecteur 6x1
    """
    newQ = q0
    newQ[1:3] = V
   # print("nouveau vecteur",newQ.shape)
    return newQ


class command():
    """
        The command object 
        for 2 DOF of yaskawa, joint 2 and joint 3 sur le plan (0,X,Z)
        for Gazebo implementation
    """

    def __init__(self):
        """ add end effector location ?"""
        print("computed torque construction")
        self.robot = RobotWrapper.BuildFromURDF(
            urdf_path, package_path, verbose=True)
        # dertermine Xinit

        self.qinit = np.array([0, np.pi/2, 0.34, 0, 0.0, 0.0])
        pin.framesForwardKinematics(
            self.robot.model, self.robot.data, self.qinit)
        self.Xinit = self.situationOT(
            self.robot.data.oMf[self.EF_index].copy())
        self.EF_index = self.robot.model.getFrameId("link_6_t")  # A CHANGER

        # determine Xc
        self.qf = np.array([0, 0, +np.pi/6, 0, 0, 0]) + \
            self.qinit  # move from 10deg from qinit
        pin.framesForwardKinematics(self.robot.model, self.robot.data, self.qf)
        self.Xf = self.situationOT(self.robot.data.oMf[self.EF_index].copy())

        # pub
        self.pubTorqueJoint1 = rospy.Publisher(
            "/motoman_hc10/joint1_torque_controller/command", Float64, queue_size=1)
        self.pubTorqueJoint2 = rospy.Publisher(
            "/motoman_hc10/joint2_torque_controller/command", Float64, queue_size=1)
        self.pubTorqueJoint3 = rospy.Publisher(
            "/motoman_hc10/joint3_torque_controller/command", Float64, queue_size=1)
        self.pubTorqueJoint4 = rospy.Publisher(
            "/motoman_hc10/joint4_torque_controller/command", Float64, queue_size=1)
        self.pubTorqueJoint5 = rospy.Publisher(
            "/motoman_hc10/joint5_torque_controller/command", Float64, queue_size=1)
        self.pubTorqueJoint6 = rospy.Publisher(
            "/motoman_hc10/joint6_torque_controller/command", Float64, queue_size=1)
        # sub

        self.measured_joint = rospy.Subscriber(
            "/motoman_hc10/joint_states", JointState, self._measured_joint_callback)  # A Adapter

        self.previousTime = rospy.get_rostime()
        self.q = reshapeVectorQTo2DOF(self.qinit)
        self.vq = np.zeros(self.qinit.shape)
        self.aq = np.zeros(self.qinit.shape)

        self.Xc = self.Xfs
        self.dXc = np.zeros(self.Xc.shape)
        self.ddXc = np.zerso(self.Xc.shape)

    def orientationEuler(self, R):
        """ Renvois l'orientation selon la valeurs des angles d'euler  
        prend une matrice de rotation 3x3 en entrée

        a changer mettre les quaternions ici ou roll pitch yaw 

        """
        #print("R22 = ",R[2,2])
        if(abs(R[2, 2]) != 1):
            psi = math.atan2(R[0, 2], -R[1, 2])
            theta = math.acos(R[2, 2])
            phi = math.atan2(R[2, 0], R[2, 1])
        else:  # attention psi et phi ne sont pas définis ici phi = 2*psi => évite la division par 0
            #print("attention psi et phi ne sont pas définis ici ils seront pris égaux")
            a = math.atan2(R[0, 1], R[0, 0])
            psi = a/(1-2*R[2, 2])
            theta = math.pi*(1-R[2, 2])/2
            phi = 2*psi
        return np.array([psi % (2*math.pi), theta % (2*math.pi), phi % (2*math.pi)])

    def situationOT(self, M):
        """ 
            cette fonction permets à partir d'un objet SE3, d'obtenir un vecteur X contenant la transaltion et la rotation de L'OT (situation de l'OT)
            avec les angles d'euler classique, M est l'objet SE3, out = [ex ey ez psi theta phi]
            degenerated this function return px and py
         """
        p = M.translation
        # delta = self.orientationRTL(M.rotation) #à decommenter a terme
        # return np.concatenate((p,delta),axis=0)
        return p[0:2]

    def _measured_joint_callback(self, data):
        """
            each time a joint as been published on the topic then we compute all data that we need to do the control law
            q,vq,aq vector 6x1
            q2dof,vq2dof,aq2dof vector 2x1
        """
        self.q = np.array(data.position)
        vqnew = np.array(data.velocity)

        self.dt = rospy.get_rostime() - self.previousTime
        self.previousTime = rospy.get_rostime()
        dt = self.dt.to_sec()
        self.aq = (vqnew-self.vq)/dt
        self.vq = vqnew

        pin.framesForwardKinematics(self.robot.model, self.robot.data, self.q)

        self.A = pin.crba(self.robot.model, self.robot.data,
                          self.q)  # compute mass matrix
        self.H = pin.rnea(self.robot.model, self.robot.data, self.q, self.vq, np.zeros(
            self.q.shape))  # compute dynamic drift -- Coriolis, centrifugal, gravity
        J6 = pin.computeFrameJacobian(self.robot.model, self.robot.data,
                                      self.q, self.EF_index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        self.J = reshapeJacobian(J6)

        q2dof = reshapeVectorQTo2DOF(self.q)
        vq2dof = reshapeVectorQTo2DOF(self.vq)
        aq2dof = reshapeVectorQTo2DOF(self.aq)

        self.Xmeasure = self.situationOT(
            self.robot.data.oMf[self.EF_index].copy())
        self.dXmeasure = np.dot(self.J, vq2dof)
        self.ddXmeasure = self.getdjv()  # compute Jdot.qdot

        self.tau = self.computedTorqueController(
            self.Xc, self.Xmeasure, self.dXc, self.dXmeasure, self.ddXc, self.ddXmeasure)
        self._publish_JointTorque()

    def computedTorqueController(self, Xc, Xm, dXc, dXm, ddXc, ddXm):
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
        kp = 1
        kd = 2*math.sqrt(kp)

        ex = Xc-Xm  # vector 2x1
        edx = dXc-dXm  # vector 2x1
        # self.error.append(np.linalg.norm(ex))
        Jp = np.linalg.pinv(self.J)  # matrix 2x2
        W = np.dot(Jp, kp*ex + kd*edx+ddXc-ddXm)  # vector 2x1
        nW = reshape2DOFToVectorQ(W, self.q.copy())
        return np.dot(self.A, nW) + self.H

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
        print("tau to publish \n ", self.tau)
        self.pubTorqueJoint1.publish(self.tau[0])
        self.pubTorqueJoint2.publish(self.tau[1])
        self.pubTorqueJoint3.publish(self.tau[2])
        self.pubTorqueJoint4.publish(self.tau[3])
        self.pubTorqueJoint5.publish(self.tau[4])
        self.pubTorqueJoint6.publish(self.tau[5])

    def getdjv(self):
        """
            this function return the product of the derivative Jacobian times the joint velocities 
        """
        djV = pin.getFrameClassicalAcceleration(
            self.robot.model, self.robot.data, self.EF_index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return djV.linear[0:2]

    def run(self):
        while(True):
            i = 0


if __name__ == "__main__":
    rospy.init_node('computed_torque_control_node')
    computed_torque = command()
    computed_torque.run()
