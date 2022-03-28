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

class Position_And_torque():
    """
        The command object 
    """
    def __init__(self):
        
        self.robot = RobotWrapper.BuildFromURDF(urdf_path,package_path,verbose=True)
        self.EF_index = self.robot.model.getFrameId("tool0") # A CHANGER
        # pub
        self.pub_Position_Joint1 = rospy.Publisher("motoman_hc10/joint1_position_controller/command",Float64,queue_size=10)
        self.pub_Position_Joint2 = rospy.Publisher("motoman_hc10/joint2_position_controller/command",Float64,queue_size=10)
        self.pub_Position_Joint3 = rospy.Publisher("motoman_hc10/joint3_position_controller/command",Float64,queue_size=10)
        self.pub_Position_Joint4 = rospy.Publisher("motoman_hc10/joint4_position_controller/command",Float64,queue_size=10)
        self.pub_Position_Joint5 = rospy.Publisher("motoman_hc10/joint5_position_controller/command",Float64,queue_size=10)
        self.pub_Position_Joint6 = rospy.Publisher("motoman_hc10/joint6_position_controller/command",Float64,queue_size=10)
        # sub
        self.measured_joint = rospy.Subscriber("/motoman_hc10/joint_states",JointState,self._measured_joint_callback) #A Adapter
        #self._joint_trajectory = rospy.Suscriber("topic",MSG,self._trajectory_callback)
        self.previousTime = rospy.get_rostime()
        self.Torque_all=[]

    def talker(self):

        dataAcc = [0, 0]
        jointNames = ['joint_1_s','joint_2_l','joint_3_u','joint_4_r','joint_5_b','joint_6_t']
        
        
        rate = rospy.Rate(10) # 10hz

        # Open file
        f = open("2dof_data_LC.txt.txt", "r")

        # robot = RobotWrapper()
        # robot.initFromURDF(urdf_path, package_path, verbose=True)
        # ROS_function(robot, ..., dataAcc)

        # A = [0,0,0,0,0,0]

        while not rospy.is_shutdown():

            # tmp = [f.readline() for i in range(1,250)]
            l = f.readline()

            # If EOF => Loop on file
            if len(l) == 0:
                f.close()
                f = open("2dof_data_LC.txt.txt", "r")
                l = f.readline()

            q_data = l.split()
            q_data_float = [float(i) for i in q_data]

            for i in range(0, 6):
                rospy.loginfo(q_data_float[i])

                self.pub_Position_Joint1.publish(q_data_float[0]) 
                self.pub_Position_Joint2.publish(q_data_float[1]) 
                self.pub_Position_Joint3.publish(q_data_float[2]) 
                self.pub_Position_Joint4.publish(q_data_float[3]) 
                self.pub_Position_Joint5.publish(q_data_float[4]) 
                self.pub_Position_Joint6.publish(q_data_float[5]) 

                rate.sleep()

    
    def _measured_joint_callback(self,data):

        self.Torque_all.append(data.effort)


if __name__ == "__main__":
    rospy.init_node('Position_And_torque')
    position_And_torque = Position_And_torque()
    position_And_torque.run()
    Q=[]
    Q=position_And_torque.Torque_all
    print("shape of Q",np.array(Q).shape)
        
        