#! /usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

def odom_callback(msg):
    # go = Odometry() is not needed
    print ("------------------------------------------------")
    print ("pose x = " + str(msg.pose.pose.position.x))
    print ("pose y = " + str(msg.pose.pose.position.y))
    print ("orientacion x = " + str(msg.pose.pose.orientation.x))
    print ("orientacion y = " + str(msg.pose.pose.orientation.y))
    rate.sleep()

def imu_callback(msg):
    # allez = Imu()
    print ("------------------------------------------------")
    print ("veloc angular z = " + str(msg.angular_velocity.z))
    print ("veloc angular y = " + str(msg.angular_velocity.y))
    print ("aceleracion linear x = " + str(msg.linear_acceleration.x))
    print ("aceleracion linear y = " + str(msg.linear_acceleration.y))
    rate.sleep()

def twist (msg):
    # move = Twist()
    print ("velocidad linear x = " + str(move.linear.x))
    print ("velocidad angular z = " + str (move.angular.z))
    rate.sleep()
    #sub=rospy.Subscriber('cmd_vel', Twist, twist)

rospy.init_node('motoman_hc10') # the original name sphero might be the same as other node.
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1) #topic publisher that allows you to move the sphero
sub_odom = rospy.Subscriber('/odom', Odometry, odom_callback) # the original name odom might be the same as other function.
sub_imu = rospy.Subscriber('/sphero/imu/data3', Imu, imu_callback)
rate = rospy.Rate(0.5)

while not rospy.is_shutdown():
    move = Twist()
    move.linear.x = 0.1 # m/s. The original value 2 is too large
    move.angular.z= 0.5 # rad/s
    pub.publish(move)
    rate.sleep() # Instead of using rospy.spin(), we should use rate.sleep because we are in a loop