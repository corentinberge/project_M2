/*
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2016, Delft Robotics Institute
 * Copyright (c) 2020, Southwest Research Institute
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *  notice, this list of conditions and the following disclaimer in the
 *  documentation and/or other materials provided with the distribution.
 *  * Neither the name of the Southwest Research Institute, nor the names
 *  of its contributors may be used to endorse or promote products derived
 *  from this software without specific prior written permission.
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

#include "motoman_driver/io_relay.h"
#include <string>
#include <cstdint>
#include <limits>
#include <ros/ros.h>
#include <sstream>

namespace motoman
{
namespace io_relay
{

using industrial::shared_types::shared_int;
using industrial::shared_types::shared_real;

bool MotomanIORelay::init(int default_port)
{
  std::string ip;
  int port;

  // override port with ROS param, if available
  const std::string port_param_name = "~port";
  // TODO( ): should really use a private NodeHandle here
  if (!ros::param::param<int>(port_param_name, port, default_port))
  {
    ROS_WARN_STREAM_NAMED("io.init", "Failed to get '" << port_param_name
      << "' parameter: using default (" << default_port << ")");
  }
  if (port < 0 || port > std::numeric_limits<uint16_t>::max())
  {
    ROS_FATAL_STREAM_NAMED("io.init", "Invalid value for port (" << port << "), "
      "must be between 0 and " << std::numeric_limits<uint16_t>::max() << ".");
    return false;
  }

  const std::string robot_ip_param_name = "robot_ip_address";
  if (!ros::param::get(robot_ip_param_name, ip) || ip.empty())
  {
    ROS_FATAL_STREAM_NAMED("io.init", "Invalid IP address: please set the '"
      << robot_ip_param_name << "' parameter");
    return false;
  }

  char* ip_addr = strdup(ip.c_str());  // connection.init() requires "char*", not "const char*"
  if (!default_tcp_connection_.init(ip_addr, port))
  {
    ROS_FATAL_NAMED("io.init", "Failed to initialize TcpClient");
    return false;
  }
  free(ip_addr);

  ROS_DEBUG_STREAM_NAMED("io.init", "I/O relay attempting to connect to: tcp://" << ip << ":" << port);
  if (!default_tcp_connection_.makeConnect())
  {
    ROS_FATAL_NAMED("io.init", "Failed to connect");
    return false;
  }

  if (!io_ctrl_.init(&default_tcp_connection_))
  {
    ROS_FATAL_NAMED("io.init", "Failed to initialize MotomanIoCtrl");
    return false;
  }

  this->srv_read_single_io = this->node_.advertiseService("read_single_io",
      &MotomanIORelay::readSingleIoCB, this);
  this->srv_write_single_io = this->node_.advertiseService("write_single_io",
      &MotomanIORelay::writeSingleIoCB, this);

  return true;
}

bool MotomanIORelay::positionCB()
{
  motoman::yrc1000_memory::Mregister::reserve reserve;
  shared_real pos_s=-1, pos_l=-1, pos_u=-1, pos_r=-1, pos_b=-1, pos_t=-1; 
  shared_real pos_x=-1, pos_y=-1, pos_z=-1; 
  shared_real rot_x=-1, rot_y=-1, rot_z=-1;

  //Axes
  this->readDoubleIO(reserve.POS_S,pos_s); this->to_deg(pos_s);
  this->readDoubleIO(reserve.POS_L,pos_l); this->to_deg(pos_l);
  this->readDoubleIO(reserve.POS_U,pos_u); this->to_deg(pos_u);
  this->readDoubleIO(reserve.POS_R,pos_r); this->to_deg(pos_r);
  this->readDoubleIO(reserve.POS_B,pos_b); this->to_deg(pos_b);
  this->readDoubleIO(reserve.POS_T,pos_t); this->to_deg(pos_t);
  
  //TCP
  this->readDoubleIO(reserve.POS_X,pos_x);  this->to_mm(pos_x);
  this->readDoubleIO(reserve.POS_Y,pos_y);  this->to_mm(pos_y);
  this->readDoubleIO(reserve.POS_Z,pos_z);  this->to_mm(pos_z);
  this->readDoubleIO(reserve.POS_Rx,rot_x); this->to_deg(rot_x);
  this->readDoubleIO(reserve.POS_Ry,rot_y); this->to_deg(rot_y);
  this->readDoubleIO(reserve.POS_Rz,rot_z); this->to_deg(rot_z);

  this->position_msg.pos_s = pos_s;
  this->position_msg.pos_l = pos_l;
  this->position_msg.pos_u = pos_u;
  this->position_msg.pos_r = pos_r;
  this->position_msg.pos_b = pos_b;
  this->position_msg.pos_t = pos_t;

  this->position_msg.pos_x = pos_x;
  this->position_msg.pos_y = pos_y;
  this->position_msg.pos_z = pos_z;

  this->position_msg.rot_x = rot_x;
  this->position_msg.rot_y = rot_y;
  this->position_msg.rot_z = rot_z;

  this->pub_position_= this->node_.advertise<motoman_msgs::Position>("joint_position",1);
  this->pub_position_.publish(this->position_msg);
}

bool MotomanIORelay::vitesseCB()
{
  motoman::yrc1000_memory::Mregister::reserve reserve;
  shared_real vit_s=-1, vit_l=-1, vit_u=-1, vit_r=-1, vit_b=-1, vit_t=-1; 
  shared_real vit_tcp=-1;

  //Axes
  this->readDoubleIO(reserve.VIT_S,vit_s); this->to_deg(vit_s);
  this->readDoubleIO(reserve.VIT_L,vit_l); this->to_deg(vit_l);
  this->readDoubleIO(reserve.VIT_U,vit_u); this->to_deg(vit_u);
  this->readDoubleIO(reserve.VIT_R,vit_r); this->to_deg(vit_r);
  this->readDoubleIO(reserve.VIT_B,vit_b); this->to_deg(vit_b);
  this->readDoubleIO(reserve.VIT_T,vit_t); this->to_deg(vit_t);
  
  //TCP
  this->readDoubleIO(reserve.VIT_TCP,vit_tcp); this->to_mm(vit_tcp);

  this->vitesse_msg.vit_s = vit_s;
  this->vitesse_msg.vit_l = vit_l;
  this->vitesse_msg.vit_u = vit_u;
  this->vitesse_msg.vit_r = vit_r;
  this->vitesse_msg.vit_b = vit_b;
  this->vitesse_msg.vit_t = vit_t;

  this->vitesse_msg.vit_tcp = vit_tcp;


  this->pub_vitesse_= this->node_.advertise<motoman_msgs::Vitesse>("joint_vitesse",1);
  this->pub_vitesse_.publish(this->vitesse_msg);
}

bool MotomanIORelay::effortCB()
{
  motoman::yrc1000_memory::Mregister::reserve reserve;
  shared_int value=-1;
  shared_real couple, force, force_totale;
  std::string err_msg;

  this->effort_msg.CoupleJoints.clear();
  this->effort_msg.CoupleTCP.clear();
  this->effort_msg.ForceTCP.clear();

  //Couple Joints
  for(int it=reserve.TRQe_S; it<=reserve.TRQe_T;it++)
  {
    this->mutex_.lock();
    bool result = this->io_ctrl_.readSingleIO(it,value,err_msg);
    this->mutex_.unlock();
    if(!result)
    {
      // provide caller with failure indication
      std::stringstream message;
      message << "Read failed (address: " << it << "): " << err_msg;
      ROS_ERROR_STREAM_NAMED("io.read", message.str());
      return result;
    }
    couple = this->to_newton(value); 
    this->effort_msg.CoupleJoints.push_back(couple);
  }

  //Couple TCP
  for(int it=reserve.TRQe_X; it<=reserve.TRQe_Z;it++)
  {
    this->mutex_.lock();
    bool result = this->io_ctrl_.readSingleIO(it,value,err_msg);
    this->mutex_.unlock();
    if(!result)
    {
      // provide caller with failure indication
      std::stringstream message;
      message << "Read failed (address: " << it << "): " << err_msg;
      ROS_ERROR_STREAM_NAMED("io.read", message.str());
      return result;
    }
    couple = this->to_newton(value); 
    this->effort_msg.CoupleTCP.push_back(couple);
  }

  //Force TCP
  for(int it=reserve.Fe_X; it<=reserve.Fe_Z;it++)
  {
    this->mutex_.lock();
    bool result = this->io_ctrl_.readSingleIO(it,value,err_msg);
    this->mutex_.unlock();
    if(!result)
    {
      // provide caller with failure indication
      std::stringstream message;
      message << "Read failed (address: " << it << "): " << err_msg;
      ROS_ERROR_STREAM_NAMED("io.read", message.str());
      return result;
    }
    force = this->to_newton(value); 
    this->effort_msg.ForceTCP.push_back(force);
  }

  //Force totale TCP
  this->mutex_.lock();
  bool result = this->io_ctrl_.readSingleIO(reserve.Fe_Totale,value,err_msg); force_totale = this->to_newton(value);
  this->mutex_.unlock();

  if(!result)
  {
    // provide caller with failure indication
    std::stringstream message;
    message << "Read failed (address: " << reserve.Fe_Totale << "): " << err_msg;
    ROS_ERROR_STREAM_NAMED("io.read", message.str());
    return result;
  }
  force_totale = this->to_newton(value);
  this->effort_msg.ForceTotaleTCP = force_totale;

  this->pub_effort_= this->node_.advertise<motoman_msgs::Effort>("joint_effort",1);
  this->pub_effort_.publish(this->effort_msg);
  return true;
}


bool MotomanIORelay::readIoCB() 
{
  motoman::yrc1000_memory::Mregister::reserve reserve;
  shared_real pos=-1;
  this->readDoubleIO(reserve.POS_X,pos);
  this->position_msg.pos_x = pos;
  this->pub_position_= this->node_.advertise<motoman_msgs::Effort>("joint_efforts",1);
  this->pub_position_.publish(this->position_msg);

  return 1;//result | result1;
}

void MotomanIORelay::readDoubleIO(shared_int address1, shared_real &myFloat)
{
    int val1 =-1;
    int val2 =-1;
    int address2 = address1 + 1;
    std::string err_msg1, err_msg2;

    this->mutex_.lock();
    bool result1 = this->io_ctrl_.readSingleIO(address1,val1,err_msg1);
    this->mutex_.unlock();
    this->mutex_.lock();
    bool result2 = this->io_ctrl_.readSingleIO(address2,val2,err_msg2);
    this->mutex_.unlock();

    std::bitset<32> p_faible(val1); //Poids faible
    std::bitset<32> p_fort(val2); //Poids fort

  if(result1 & result2)
  {
    if(val2 != 0)
    {
      //Décalage du poids fort. 2^(n) - 1 avec n=16.  
      p_fort = (p_fort<<16);
      
      //Concatenation et conversion en ulong
      unsigned long conv = (p_fort | p_faible).to_ulong(); 
      
      ////////////////////
      // Complement à 2 //
      ////////////////////

      //Inversion des bits
      std::bitset<32> bs200((p_fort | p_faible).flip());
      //conversion en ulong
      unsigned long convFlip = bs200.to_ulong(); 
      //Addition (+1)
      int a = 1;      
      std::bitset<32> bin_Ca2(convFlip+a);
      
      //Conversion en ulong après le Complement a 2.
      unsigned long conv_Ca2 = bin_Ca2.to_ulong();

      if(val2>=32768)   //Valeur négative.  1<<15 = 32768. 
      {
        myFloat = conv_Ca2;
        myFloat = -1*myFloat;
      }
      else //Valeur positive.
      {
        myFloat = (float)conv;
      }
    }
    else //p_fort = 0
    {
      //Conversion en ulong
      unsigned long conv = p_faible.to_ulong();
      myFloat = (float)conv;
    }

  }
  else
  {
    if(!result1)
    {
      // provide caller with failure indication
      std::stringstream message;
      message << "Read failed (address: " << address1 << "): " << err_msg1;
      ROS_ERROR_STREAM_NAMED("io.read", message.str());
    }
    
    if(!result2)
    {
      // provide caller with failure indication
      std::stringstream message;
      message << "Read failed (address: " << address2 << "): " << err_msg1;
      ROS_ERROR_STREAM_NAMED("io.read", message.str());
    }
  }

}

shared_real MotomanIORelay::to_newton(shared_int in)
{
  shared_real out;
  return out = (in - 10000)*0.1;
}

void MotomanIORelay::to_mm(shared_real &value)
{
  value = value*1e-3;
}

void MotomanIORelay::to_deg(shared_real&value)
{
  value = value*1e-4;
}

// Service to read a single IO
bool MotomanIORelay::readSingleIoCB(
  motoman_msgs::ReadSingleIO::Request &req,
  motoman_msgs::ReadSingleIO::Response &res)
{
  shared_int io_val = -1;
  std::string err_msg;

  // send message and release mutex as soon as possible
  this->mutex_.lock();
  bool result = io_ctrl_.readSingleIO(req.address, io_val, err_msg);
  this->mutex_.unlock();

  if (!result)
  {
    res.success = false;

    // provide caller with failure indication
    // TODO( ): should we also return the result code?
    std::stringstream message;
    message << "Read failed (address: " << req.address << "): " << err_msg;
    res.message = message.str();
    ROS_ERROR_STREAM_NAMED("io.read", res.message);

    return true;
  }

  ROS_DEBUG_STREAM_NAMED("io.read", "Address " << req.address << ", value: " << io_val);

  // no failure, so no need for an additional message
  res.value = io_val;
  res.success = true;
  return true;
}


// Service to write Single IO
bool MotomanIORelay::writeSingleIoCB(
  motoman_msgs::WriteSingleIO::Request &req,
  motoman_msgs::WriteSingleIO::Response &res)
{
  std::string err_msg;

  // send message and release mutex as soon as possible
  this->mutex_.lock();
  bool result = io_ctrl_.writeSingleIO(req.address, req.value, err_msg);
  this->mutex_.unlock();

  if (!result)
  {
    res.success = false;

    // provide caller with failure indication
    // TODO( ): should we also return the result code?
    std::stringstream message;
    message << "Write failed (address: " << req.address << "): " << err_msg;
    res.message = message.str();
    ROS_ERROR_STREAM_NAMED("io.write", res.message);

    return true;
  }

  ROS_DEBUG_STREAM_NAMED("io.write", "Element " << req.address << " set to: " << req.value);

  // no failure, so no need for an additional message
  res.success = true;
  return true;
}

}  // namespace io_relay
}  // namespace motoman

