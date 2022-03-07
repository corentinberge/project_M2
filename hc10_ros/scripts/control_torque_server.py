#!/usr/bin/env python

from __future__ import print_function

from hc10_ros.srv import joint_state
import rospy

def handle_control_torque(req):
    print("Returning %s %s %s %s\n"%(req.name, req.position, req.velocity, req.effort))
    return ControlTorqueResponse(req.name, req.position, req.velocity, req.effort)

def control_torque_server():
    rospy.init_node('control_torque_server')
    s = rospy.Service('control_torque', ControlTorque, handle_control_torque)
    print("Ready to control torque")
    rospy.spin()

if __name__ == "__main__":
    control_torque_server()