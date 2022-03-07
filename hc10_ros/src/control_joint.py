#!/usr/bin/env python
import roslib;
import rospy, yaml, sys
from hc10_ros.msg import JointCommands
from hc10_ros.msg import JointState
from numpy import zeros, array, linspace
from math import ceil

JointNames = ['joint_1_s', 'joint_2_l', 'joint_3_u', 'joint_4_r','joint_5_b', 'joint_6_t']

currentJointState = JointState()
def jointStatesCallback(msg):
  global currentJointState
  currentJointState = msg

if __name__ == '__main__':
  # first make sure the input arguments are correct

  #Mode lecture fichier
  # if len(sys.argv) != 3:
  #   print "usage: traj_yaml.py YAML_FILE TRAJECTORY_NAME"
  #   print "  where TRAJECTORY is a dictionary defined in YAML_FILE"
  #   sys.exit(1)
  
  
  # traj_yaml = yaml.load(file(sys.argv[1], 'r'))
  # traj_name = sys.argv[2]

  
  # if not traj_name in traj_yaml:
  #   print "unable to find trajectory %s in %s" % (traj_name, sys.argv[1])
  #   sys.exit(1)
  # traj_len = len(traj_yaml[traj_name])




  #Mode test
  traj_yaml = [[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]]
  traj_len = len(traj_yaml)

  # Setup subscriber to atlas states
  rospy.Subscriber("/motoman_hc10/joint_states", JointState, jointStatesCallback)

  # initialize JointCommands message
  command = JointCommands()
  command.name = list(JointNames)
  n = len(command.name)
  command.position     = zeros(n)
  command.velocity     = zeros(n)
  command.effort       = zeros(n)
  command.kp_position  = zeros(n)
  command.ki_position  = zeros(n)
  command.kd_position  = zeros(n)
  command.kp_velocity  = zeros(n)
  command.i_effort_min = zeros(n)
  command.i_effort_max = zeros(n)

  # now get gains from parameter server
  rospy.init_node('motoman_hc10_control')
  for i in xrange(len(command.name)):
    name = command.name[i]
    # pid = rospy.get_param('/motoman_hc10/joint' + str(i+1) + '_position_controller/pid/parameter_descriptions')
    pid = [1,2,3]
    # print("\n\n---PID---\n"+pid+"\n\---END PID---\n\n")
    command.kp_position[i]  = pid[0]
    command.ki_position[i]  = pid[1]
    command.kd_position[i]  = pid[2]
    command.i_effort_max[i] = 1.0
    command.i_effort_min[i] = -1
    # command.i_effort_max[i] = rospy.get_param('motoman_hc10/gains/' + name[7::] + '/i_clamp')
    # command.i_effort_min[i] = -command.i_effort_max[i]
    

  # set up the publisher
  pub = rospy.Publisher('/motoman_hc10/joint_commands', JointCommands, queue_size=1)

  # for each trajectory
  for i in xrange(0, traj_len):
    # get initial joint positions
    initialPosition = array(currentJointState.position)
    # get joint commands from yaml
    #y = traj_yaml[traj_name][i]
    y = traj_yaml[i]
    # first value is time duration
    dt = float(1)
    # subsequent values are desired joint positions
    # commandPosition = array([ float(x) for x in y.split() ])
    commandPosition = y
    # desired publish interval
    dtPublish = 0.02
    n = ceil(dt / dtPublish)
    for ratio in linspace(0, 1, n):
      interpCommand = (1-ratio)*initialPosition 
      interpCommand += int(ratio) * commandPosition
      # interpCommand = (1-ratio)*initialPosition + ratio * commandPosition
      # command.position = [ float(x) for x in interpCommand ]
      # pub.publish(command)
      # rospy.sleep(dt / float(n))