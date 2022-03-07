/*
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2020, Southwest Research Institute
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *       * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *       * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *       * Neither the name of the Southwest Research Institute, nor the names
 *       of its contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef MOTOMAN_DRIVER_IO_RELAY_H
#define MOTOMAN_DRIVER_IO_RELAY_H

#include "simple_message/socket/tcp_client.h"
#include "motoman_driver/io_ctrl.h"
#include "motoman_msgs/ReadSingleIO.h"
#include "motoman_msgs/WriteSingleIO.h"
#include <boost/thread.hpp>

#include "motoman_msgs/Position.h"
#include "motoman_msgs/Vitesse.h"
#include "motoman_msgs/Effort.h"
#include "motoman_driver/motoman_memory.h"
#include <bitset>

//using motoman::yrc1000_memory::Mregister;
using industrial::shared_types::shared_int;
using industrial::shared_types::shared_real;


namespace motoman
{
namespace io_relay
{

using industrial::tcp_client::TcpClient;

/**
 * \brief Message handler that sends I/O service requests to the robot controller and receives the responses.
 */
class MotomanIORelay
{
public:
  /**
   * \brief Class initializer
   *
   * \param default_port
   * \return true on success, false otherwise
   */
  bool init(int default_port);

  /**
   * \brief Read registers
   *
   * \param default_port
   * \return true on success, false otherwise
   */
  bool readIoCB();

  bool positionCB();
  bool vitesseCB();
  bool effortCB();
  shared_real to_newton(shared_int in);
  void to_mm(shared_real &value);
  void to_deg(shared_real &value);

  void readDoubleIO(shared_int address1, shared_real &myFloat);

protected:
  io_ctrl::MotomanIoCtrl io_ctrl_;
  motoman_msgs::Position position_msg;
  motoman_msgs::Vitesse vitesse_msg;
  motoman_msgs::Effort effort_msg;

  ros::ServiceServer srv_read_single_io;   // handle for read_single_io service
  ros::ServiceServer srv_write_single_io;   // handle for write_single_io service
  ros::Publisher pub_position_;
  ros::Publisher pub_vitesse_;
  ros::Publisher pub_effort_;


  ros::NodeHandle node_;
  boost::mutex mutex_;
  TcpClient default_tcp_connection_;

  bool readSingleIoCB(motoman_msgs::ReadSingleIO::Request &req,
                            motoman_msgs::ReadSingleIO::Response &res);
  bool writeSingleIoCB(motoman_msgs::WriteSingleIO::Request &req,
                            motoman_msgs::WriteSingleIO::Response &res);
};

}  // namespace io_relay
}  // namespace motoman

#endif  // MOTOMAN_DRIVER_IO_RELAY_H
