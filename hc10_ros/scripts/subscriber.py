import rospy
# from std_msgs.msg import String
from sensor_msgs.msg import JointState

f = open("data_torque.txt", "w")

def callback(data):
    # pos = data.position
    position_float = [float(i) for i in data.position]
    # efforts = data.effort
    efforts_float = [float(i) for i in data.effort]
    # print(efforts_float)
    f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
    position_float[0],position_float[1],position_float[2],position_float[3],
    position_float[4],position_float[5],efforts_float[0],efforts_float[1],
    efforts_float[2],efforts_float[3],efforts_float[4],efforts_float[5]))
    # rospy.loginfo("I heard %s",data.data)
    
def listener():
    rospy.init_node('sub')  
    rospy.Subscriber('/motoman_hc10/joint_states', JointState, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    
if __name__ == '__main__':
    # f = open("data_torque.txt", "w")
    listener()