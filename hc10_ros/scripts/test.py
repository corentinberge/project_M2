#! /usr/bin/env python
from os import wait
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('execute_trajectory',anonymous=True)

#Misc variables
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
group = moveit_commander.MoveGroupCommander("Yaskawa_arm")

group_state = group.get_named_targets()

# group.set_joint_value_target([-0.959931,-0.314159,1.69297,0.05,-1.98968,0.959931])
# group.plan()
group.go([-0.959931,-0.314159,1.69297,0.05,-1.98968,0.959931])
    