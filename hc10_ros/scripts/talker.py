#!/usr/bin/env python
# license removed for brevity

import rospy
from subscriber import *

from std_msgs.msg import Float64

def talker():
    dataAcc = [0, 0]
    jointNames = ['joint_1_s','joint_2_l','joint_3_u','joint_4_r','joint_5_b','joint_6_t']
    pub = []
    for i in range(0, 6):
        pub.append(rospy.Publisher('/motoman_hc10/joint'+str(i+1)+'_position_controller/command', Float64, queue_size=10))
        rospy.init_node('talker', anonymous=True)
        rate = rospy.Rate(100) # 10hz

    # Open file
    f = open("data_speed.txt", "r")
    l = f.readline()
    

    while not (rospy.is_shutdown() and len(l) != 0):
            
        # tmp = [f.readline() for i in range(1,50)]
        l = f.readline()
        if(len(l) == 0):
            break

        # If EOF => Loop on file
        # if len(l) == 0:
        #     f.close()
        #     f = open("data_speed.txt", "r")
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
    print('Done!')
    exit(0)
