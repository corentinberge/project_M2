ls""" 
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

# pinocchio 
import pinocchio as pin
from pinocchio.utils import *
from pinocchio.robot_wrapper import RobotWrapper

#ros 
import rospy

#ROS MSG
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import ApplyJointEffort
from std_msgs.msg import Float64

package_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + '/src/hc10_ros'
urdf_path = package_path + '/motoman_hc10_support/urdf/hc10_gazebo.urdf'

class command():
    """
        The command object 
    """
    def __init__(self):
        """ add end effector location ?"""
        print("computed torque construction")
        self.robot = RobotWrapper.BuildFromURDF(urdf_path,package_path,verbose=True)
        self.EF_index = self.robot.model.getFrameId("tool0") # A CHANGER
        # pub
        self.pubTorqueJoint1 = rospy.Publisher("motoman_hc10/joint_1_s_torque_controller/command",Float64,queue_size=1)
        self.pubTorqueJoint2 = rospy.Publisher("motoman_hc10/joint_2_l_torque_controller/command",Float64,queue_size=1)
        self.pubTorqueJoint3 = rospy.Publisher("motoman_hc10/joint_3_u_torque_controller/command",Float64,queue_size=1)
        self.pubTorqueJoint4 = rospy.Publisher("motoman_hc10/joint_4_r_torque_controller/command",Float64,queue_size=1)
        self.pubTorqueJoint5 = rospy.Publisher("motoman_hc10/joint_5_b_torque_controller/command",Float64,queue_size=1)
        self.pubTorqueJoint6 = rospy.Publisher("motoman_hc10/joint_6_t_torque_controller/command",Float64,queue_size=1)
        # sub
        self.measured_joint = rospy.Subscriber("/motoman_hc10/joint_states",JointState,self._measured_joint_callback) #A Adapter
        #self._joint_trajectory = rospy.Suscriber("topic",MSG,self._trajectory_callback)
        self.previousTime = rospy.get_rostime()
        self.aq = 0
        self.vq = 0
        self.dt = 0

    def orientationEuler(self,R):
        """ Renvois l'orientation selon la valeurs des angles d'euler  
        prend une matrice de rotation 3x3 en entrée"""
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
        return np.array([psi,theta,phi])

    def situationOT(self,M):
        """ cette fonction permets à partir d'un objet SE3, d'obtenir un vecteur X contenant la transaltion et la rotation de L'OT (situation de l'OT)
        avec les angles d'euler classique, M est l'objet SE3, out = [ex ey ez psi theta phi] """
        p = M.translation
        delta = self.orientationEuler(M.rotation) #à decommenter a terme 
        return np.concatenate((p,delta),axis=0)
    
    def _measured_joint_callback(self,data):
        """
            each time a joint as been published on the topic then we compute all data that we need to do the control law
        """
        self.q = np.array(data.position)
        vqnew = np.array(data.velocity)
        #print("position ",self.q)
        #print("velocity ",vqnew)
        self.dt = rospy.get_rostime() - self.previousTime
        dt = self.dt.to_sec()
        self.aq = (vqnew-self.vq)/dt
        self.vq = vqnew

        self.robot.forwardKinematics(self.q,self.vq,0*self.aq)
        pin.updateFramePlacements(self.robot.model,self.robot.data)


        self.Xc = self.situationOT(self.robot.data.oMf[self.EF_index])
        self.dXc = self.Xc
        self.ddXc = self.dXc # A enlever quand on auras trouvé les trajectoires 

        self.J = pin.computeFrameJacobian(self.robot.model,self.robot.data,self.q,self.EF_index,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        self.A = pin.crba(self.robot.model,self.robot.data,self.q) # compute mass matrix
        self.H = pin.rnea(self.robot.model,self.robot.data,self.q,self.vq,np.zeros(self.q.shape))  # compute dynamic drift -- Coriolis, centrifugal, gravity
        self.getdjv() # we compute ddXmeasure
        self.Xmeasure = self.situationOT(self.robot.data.oMf[self.EF_index])
        self.dXmeasure = self.J@self.q
        self.tau = self.computedTorqueController(self.Xc,self.Xmeasure,0*self.dXc,self.dXmeasure,0*self.ddXc,self.ddXmeasure) # mettre valeur de Xc 
        self._publish_JointTorque() 
        
    
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
        kp=1
        kd = 2*math.sqrt(kp)
        ex = Xc-Xm
        edx = dXc-dXm
        Jp = np.linalg.pinv(self.J)
        W= kp*ex + kd*edx+ddXc-ddXm
        jpw = np.dot(Jp,W)
        return np.dot(self.A,jpw) + self.H

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
        print("publishing torque")
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
        djV = pin.getFrameClassicalAcceleration(self.robot.model,self.robot.data,self.EF_index,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        self.ddXmeasure = np.hstack((djV.linear,djV.angular))

    def run(self):
        while(True):
            i = 0
        



    





if __name__ == "__main__":
    rospy.init_node('computed_torque_control_node')
    computed_torque = command()
    computed_torque.run()
