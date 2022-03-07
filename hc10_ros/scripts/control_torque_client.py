#!/usr/bin/env python

from __future__ import print_function

import sys
import rospy
from hc10_ros.srv import joint_state

def control_torque_client(position,velocity,effort):
    rospy.wait_for_service('control_torque')
    try:
        control_torque = rospy.ServiceProxy('control_torque', ControlTorque)
        #Traitement
        

def usage():
    return "%s [name position velocity effort]"%sys.argv[0]

if __name__ == "__main__":
    if len(sys.argv) == 4:
        name = str(sys.argv[1])
        position = float(sys.argv[2])
        velocity = float(sys.argv[3])
        effort = float(sys.argv[4])
    else:
        print(usage())
        sys.exit(1)
    control_torque_client()
    print("%s : %s %s %s"%(name,position,velocity,effort))