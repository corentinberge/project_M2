#! /usr/bin/env python
from os import wait
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

# group.set_joint_value_target([-0.959931,-0.314159,1.69297,0.05,-1.98968,0.959931])
# group.plan()
#roup.go([-0.959931,-0.314159,1.69297,0.05,-1.98968,0.959931])

def talker():

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('execute_trajectory',anonymous=True)
    rate = rospy.Rate(100)

    #Misc variables
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander("Yaskawa_arm")

    group_state = group.get_named_targets()

    # Open file
    f = open("data/trajectoire_simple.txt", "r")
    l = f.readline()

    while not rospy.is_shutdown():

	# while(len(l) != 0):  
	tmp = [f.readline() for i in (1,50)]
	l = f.readline()

	q_data = l.split()
	q_data_float = [float(i) for i in q_data]

		# group.plan(q_data_float)
        group.go(q_data_float)
        
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
