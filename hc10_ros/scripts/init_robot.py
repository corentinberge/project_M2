import rospy
from subscriber import *

from std_msgs.msg import Float64


pos = [-0.959931, -0.314159, 1.69297, 0.05, -1.98968, 0.959931]

pub = []
for i in range(0, 6):
    pub.append(rospy.Publisher('/motoman_hc10/joint'+str(i+1)+'_position_controller/command', Float64, queue_size=10))
    rospy.init_node('init', anonymous=True)
    rate = rospy.Rate(100) # 100hz

for i in range(0, 6):
    pub[i].publish(pos[i])
