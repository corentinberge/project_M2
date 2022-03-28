#! /usr/bin/env python
from os import wait
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import time

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('execute_trajectory',anonymous=True)

#Misc variables
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
group = moveit_commander.MoveGroupCommander("Yaskawa_arm")

group_state = group.get_named_targets()

# group.set_joint_value_target([-0.959931,-0.314159,1.69297,0.05,-1.98968,0.959931])
# group.plan()

start_time = time.time()
group.set_max_velocity_scaling_factor(0.5)
group.set_joint_value_target(group.get_named_target_values("pos_to_chandelle"))
group.go(wait=True)
group.set_joint_value_target(group.get_named_target_values("pos_to_chandelle_2"))
group.go(wait=True)
group.set_max_velocity_scaling_factor(1)
group.set_joint_value_target(group.get_named_target_values("pos_to_chandelle"))
group.go(wait=True)
group.set_joint_value_target(group.get_named_target_values("pos_to_chandelle_2"))
group.go(wait=True)

group.set_joint_value_target(group.get_named_target_values("test_art_2"))
group.go(wait=True)
group.set_max_velocity_scaling_factor(0.5)
group.set_joint_value_target(group.get_named_target_values("test_art_2_2"))
group.go(wait=True)
group.set_joint_value_target(group.get_named_target_values("test_art_2"))
group.go(wait=True)
group.set_max_velocity_scaling_factor(1)
group.set_joint_value_target(group.get_named_target_values("test_art_2_2"))
group.go(wait=True)
group.set_joint_value_target(group.get_named_target_values("test_art_2"))
group.go(wait=True)

group.set_joint_value_target(group.get_named_target_values("pos_chandelle_2"))
group.go(wait=True)
group.set_max_velocity_scaling_factor(0.5)
group.set_joint_value_target(group.get_named_target_values("test_art_3"))
group.go(wait=True)
group.set_joint_value_target(group.get_named_target_values("test_art_3_2"))
group.go(wait=True)
group.set_max_velocity_scaling_factor(1)
group.set_joint_value_target(group.get_named_target_values("test_art_3"))
group.go(wait=True)
group.set_joint_value_target(group.get_named_target_values("test_art_3_2"))
group.go(wait=True)

group.set_joint_value_target(group.get_named_target_values("pos_chandelle_2"))
group.set_max_velocity_scaling_factor(0.5)
group.set_joint_value_target(group.get_named_target_values("test_art_4"))
group.go(wait=True)
group.set_joint_value_target(group.get_named_target_values("test_art_4_2"))
group.go(wait=True)
group.set_max_velocity_scaling_factor(1)
group.set_joint_value_target(group.get_named_target_values("test_art_4"))
group.go(wait=True)
group.set_joint_value_target(group.get_named_target_values("test_art_4_2"))
group.go(wait=True)

group.set_joint_value_target(group.get_named_target_values("pos_chandelle_2"))
group.set_max_velocity_scaling_factor(0.5)
group.set_joint_value_target(group.get_named_target_values("test_art_5"))
group.go(wait=True)
group.set_joint_value_target(group.get_named_target_values("test_art_5_2"))
group.go(wait=True)
group.set_max_velocity_scaling_factor(1)
group.set_joint_value_target(group.get_named_target_values("test_art_5"))
group.go(wait=True)
group.set_joint_value_target(group.get_named_target_values("test_art_5_2"))
group.go(wait=True)

group.set_joint_value_target(group.get_named_target_values("pos_chandelle_2"))
group.set_max_velocity_scaling_factor(0.5)
group.set_joint_value_target(group.get_named_target_values("test_art_6"))
group.go(wait=True)
group.set_joint_value_target(group.get_named_target_values("test_art_6_2"))
group.go(wait=True)
group.set_max_velocity_scaling_factor(1)
group.set_joint_value_target(group.get_named_target_values("test_art_6"))
group.go(wait=True)
group.set_joint_value_target(group.get_named_target_values("test_art_6_2"))
group.go(wait=True)

group.set_joint_value_target(group.get_named_target_values("pos_chandelle_2"))
group.set_max_velocity_scaling_factor(0.5)
group.set_joint_value_target(group.get_named_target_values("test_art_4_6"))
group.go(wait=True)
group.set_joint_value_target(group.get_named_target_values("test_art_6_4"))
group.go(wait=True)
group.set_max_velocity_scaling_factor(1)
group.set_joint_value_target(group.get_named_target_values("test_art_4_6"))
group.go(wait=True)
group.set_joint_value_target(group.get_named_target_values("test_art_6_4"))
group.go(wait=True)
print("--- %s seconds ---" % (time.time() - start_time))