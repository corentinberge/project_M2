""" 
    This file contain the command_node
    2 suscriber : 
        1 for the trajectory node
        1 for the Simulation node (the robot)
    1 publisher :
        1 for the simulation node the robot
"""




import queue
import pinocchio as pin
import os
from pinocchio.utils import *
from pinocchio.visualize import GepettoVisualizer
from pinocchio.robot_wrapper import RobotWrapper
import rospy

package_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + '/Modeles/'
urdf_path = package_path + 'motoman_hc10_support/urdf/hc10.urdf'

class command():
    """
        The command object 
    """
    
    def __init__(self):
        self.robot = RobotWrapper.BuildFromURDF(urdf_path,package_path,verbose=True)
        self.EF_index = robot.model.getFrameId("Nom END EFFECTOR") # A CHANGER
        # pub
        self.JointPub = rospy.Publisher("topic",JointState,queue_size=1)
        # sub
        self.measured_joint = rospy.Subscriber("topic",JointState,self._measured_joint_callback) #A Adapter
        self._joint_trajectory = rospy.Suscriber("topic",MSG,self._joint_trajectory_callback)
        self.previousTime = rospy.get_rostime()
        self.dt = 0

    def orientationEuler(self,R):
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
        self.q = data.position
        self.vq = data.velocity
        self.aq = data.acceleration
        self.dt = rospy.get_rostime() - self.previousTime
        self.robot.forwardKinematics(self.q)
        pin.updateFramePlacement(self.robot.model,self.robot.data)
        self.J = pin.ComputeFrameJacobian(self.robot.model,self.robot.data,self.q,self.EF_index,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        self.A = pin.crba(self.robot.model,self.robot.data,self.q) # compute mass matrix
        self.H = pin.rnea(self.robot.model,self.robot.data,q,self.vq,np.zeros(self.q.shape))  # compute dynamic drift -- Coriolis, centrifugal, gravity
        self.jdv = self.getdjv()
        self.X = self.situationOT(self.robot.data.omF[self.EF_index])


    def _joint_trajectory_callback(self,data):
        """ 
            trajectory node, send a message every x seconde
        """
        self.Xc = data.EFPosition
        self.dXc = data.EFVelocity
        self.ddXc = data.EFAcceleration

    def _publishQ(self):
        """
            put the control law here 
        """
        
        

        
    def getdjv(self):
        """
            this function return the product of the derivative Jacobian times the joint velocities 
        """ 
        djV = pin.getFrameClassicalAcceleration(self.robot.model,self.robot.data,self.EF_index,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        self.dJv = np.hstack(djV.linear,djV.rotation)





    





if __name__ == "__main__":
    listener = rospy.Subscriber("nom_topic",msg,callback)
    q = listener.data