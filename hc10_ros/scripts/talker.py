#!/usr/bin/env python
# license removed for brevity

import rospy

from std_msgs.msg import Float64
# from pinocchio.robot_wrapper import RobotWrapper
# from Fonction_jo import ROS_function

def talker():
    dataAcc = [0, 0]
    jointNames = ['joint_1_s','joint_2_l','joint_3_u','joint_4_r','joint_5_b','joint_6_t']
    pub = []
    for i in range(0, 6):
        pub.append(rospy.Publisher('/motoman_hc10/joint'+str(i+1)+'_position_controller/command', Float64, queue_size=10))
        rospy.init_node('talker', anonymous=True)
        rate = rospy.Rate(10) # 10hz

    # Open file
    f = open("data/trajectoire_simple.txt", "r")

    # robot = RobotWrapper()
    # robot.initFromURDF(urdf_path, package_path, verbose=True)
    # ROS_function(robot, ..., dataAcc)

    # A = [0,0,0,0,0,0]

    while not rospy.is_shutdown():
            
        # tmp = [f.readline() for i in range(1,250)]
        l = f.readline()

        # If EOF => Loop on file
        # if len(l) == 0:
        #     f.close()
        #     f = open("trajectoire.txt", "r")
        #     l = f.readline()

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
