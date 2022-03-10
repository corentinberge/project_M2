#!/usr/bin/env python
# license removed for brevity

import rospy
import os

from std_msgs.msg import Float64
from pinocchio.robot_wrapper import RobotWrapper
from Fonction_jo import ROS_function

def talker():
    Hz = 10 # 10hz
    position = 0
    velocity = 0
    acceleration = 0
    dt = 1 / Hz

    package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = package_path + '/urdf/hc10.urdf'

    jointNames = ['joint_1_s','joint_2_l','joint_3_u','joint_4_r','joint_5_b','joint_6_t']
    
    pub = []
    for i in range(0, 6):
        pub.append(rospy.Publisher('/motoman_hc10/joint'+str(i+1)+'_position_controller/command', Float64, queue_size=10))
        rospy.init_node('talker', anonymous=True)
        rate = rospy.Rate(Hz)

    robot = RobotWrapper()
    robot.initFromURDF(urdf_path, package_path, verbose=True)

    [pos, vel, acc] = ROS_function(robot, position, velocity, acceleration, 10, 0, 0, dt)
    print('{} \t {} \t {}'.format(pos, vel, acc))
    # print(pos+"\n"+vel+"\n"+acc+"\n")

    while not rospy.is_shutdown():   
        tmp = [f.readline() for i in range(1,250)] # Pour aller plus vite on skip des lignes
        q_data = l.split()
        q_data_float = [float(i) for i in q_data]

        for i in range(0, 6):
            rospy.loginfo(q_data_float[i])
            pub[i].publish(q_data_float[i])
        
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
