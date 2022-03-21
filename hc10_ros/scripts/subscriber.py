#!/usr/bin/env python

import sys, getopt
import rospy
# from std_msgs.msg import String
from sensor_msgs.msg import JointState
from motoman_msgs.msg import Position, Vitesse, Effort
from math import pi

f_position, f_vitesse, f_effort, f_all = []

def callback_all(data):
    position_float = [float(data.position[i]) for i in range(6)]
    vitesse_float  = [float(data.velocity[i]) for i in range(6)]
    effort_float   = [float(data.effort[i]) for i in range(6)]
    f_all.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
        position_float[0], position_float[1], position_float[2], position_float[3], position_float[4], position_float[5],
        vitesse_float[0], vitesse_float[1], vitesse_float[2], vitesse_float[3], vitesse_float[4], vitesse_float[5],
        effort_float[0], effort_float[1], effort_float[2], effort_float[3], effort_float[4], effort_float[5],
    ))


def callback_position(data):
    # pos = data.position
    position_float = [data.pos_s,data.pos_l,data.pos_u,data.pos_r,data.pos_b,data.pos_t]
    f_position.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
        position_float[0]*pi/180, position_float[1]*pi/180, position_float[2]*pi/180, position_float[3]*pi/180, position_float[4]*pi/180, position_float[5]*pi/180
    ))
    # rospy.loginfo("I heard %s",data.data)

def callback_vitesse(data):
    # pos = data.position
    vitesse_float = [data.vit_s,data.vit_l,data.vit_u,data.vit_r,data.vit_b,data.vit_t]
    f_vitesse.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
        vitesse_float[0], vitesse_float[1], vitesse_float[2], vitesse_float[3], vitesse_float[4], vitesse_float[5]
    ))

def callback_effort(data):
    # pos = data.position
    effort_float = [float(data.CoupleJoints[i]) for i in range(6)]
    f_effort.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
        effort_float[0], effort_float[1], effort_float[2], effort_float[3], effort_float[4], effort_float[5]
    ))

def listener(argv):

    moveit = False

    try:
        opts, args = getopt.getopt(argv,'m',['mv'])
    except getopt.GetoptError:
        print('Usage: python subscripber.py [--mv]')
        sys.exit(2)

    for opt, args in opts:
        if opt == '-h':
            print('Usage: python subscripber.py [-mv]')
            sys.exit()
        elif opt in ('-m', '--mv'):
            moveit = True

    rospy.init_node('sub')
    
    if moveit:
        print('Subscribing to Moveit topics')
        f_position = open("data/data_position.txt", "w")
        f_vitesse = open("data/data_vitesse.txt", "w")
        f_effort = open("data/data_effort.txt", "w")
        rospy.Subscriber('/joint_position', Position, callback_position)
        rospy.Subscriber('/joint_vitesse', Vitesse, callback_vitesse)
        rospy.Subscriber('/joint_effort', Effort, callback_effort)
    else:
        print('Subscribing to Gazebo topics')
        f_all = open("data/data_all.txt", "w")
        rospy.Subscriber('/motoman_hc10/joint_states', JointState, callback_all)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    
if __name__ == '__main__':
    # f = open("data_torque.txt", "w")
    listener(sys.argv[1:])
