#include <math.h>
#include <ros/ros.h>
#include <ros/subscribe_options.h>
#include <boost/thread.hpp>
#include <boost/algorithm/string.hpp>
#include <sensor_msgs/JointState.h>
#include <osrf_msgs/JointCommands.h>

ros::Publisher pub_joint_commands_;
osrf_msgs::JointCommands jointcommands;

void SetJointStates(const sensor_msgs::JointState::ConstPtr &_js)
{
  static ros::Time startTime = ros::Time::now();
  {
    // for testing round trip time
    jointcommands.header.stamp = _js->header.stamp;

    // assign sinusoidal joint angle targets
    for (unsigned int i = 0; i < jointcommands.name.size(); i++)
      jointcommands.position[i] =
        3.2* sin((ros::Time::now() - startTime).toSec());

    pub_joint_commands_.publish(jointcommands);
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "test");
  ros::NodeHandle n;

  //ros::ServiceServer service = n.advertiseService("server", return_joint);
  ROS_INFO("Ready to return joint.");
  ros::spin();


  printf("%s : %f %f %f",);

  return 0;
}