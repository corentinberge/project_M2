from cProfile import label
from numpy.linalg.linalg import det, transpose
import pinocchio as pin
from pinocchio.utils import *
from pinocchio.visualize import GepettoVisualizer
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import norm, inv, pinv
from scipy.linalg import pinv2
from pathlib import Path
import pandas as pd
import time
import os
import csv

import rospy                            # For node ROS
from std_msgs.msg import String

def situationOT(M):
    """ cette fonction permets à partir d'un objet SE3, d'obtenir un vecteur X contenant la transaltion et la rotation de L'OT (situation de l'OT)
    avec les angles d'euler classique, M est l'objet SE3, out = [ex ey ez psi theta phi] """
    p = M.translation
    delta = orientationEuler(M.rotation)
    return np.concatenate((p,delta),axis=0)


def orientationEuler(R):
    """ Renvois l'orientation selon la valeurs des angles d'euler  
    prend une matrice de rotation 3x3 en entrée"""
    if(abs(R[2,2]) != 1):
        psi = math.atan2(R[0,2],-R[1,2])
        theta = math.acos(R[2,2])
        phi = math.atan2(R[2,0],R[2,1])
    else : # attention psi et phi ne sont pas définis ici phi = 2*psi => évite la division par 0 
        #print("attention psi et phi ne sont pas définis ici ils seront pris égaux")
        a = math.atan2(R[0,1],R[0,0])
        psi = a/(1-2*R[2,2])
        theta = math.pi*(1-R[2,2])/2
        phi = 2*psi
    return np.array([psi,theta,phi])

def loiPendule(robot,t):
    """retourne la loi avec série de fournier """
    q = []
    dq = []
    ddq = []
    for i in range(robot.nq):
        q.append(0.5*np.cos(2*math.pi*t))
        dq.append(-1*math.pi*np.sin(2*math.pi*t))
        ddq.append(-2*math.pi**2*np.cos(2*math.pi*t))
    return np.array(q),np.array(dq),np.array(ddq)

def getdjv(robot,q,v,a):
    """this function return the product of the derivative Jacobian times the joint velocities """ 
    IDX = robot.model.getFrameId("tool0")
    robot.forwardKinematics(q,v,0*a)
    dJv = np.hstack( (pin.getFrameClassicalAcceleration(robot.model,robot.data,IDX,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear ,pin.getFrameAcceleration(robot.model,robot.data,IDX,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).angular))
    robot.forwardKinematics(q,v,a)
    return dJv

def computePlanarJacobian(robot,q,IDX):
    """
            this function compute the planar jacobian for the robot, in the reference frame 
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    """
    J = pin.computeFrameJacobian(robot.model,robot.data,q,IDX,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    return J

def getRobot():
    """ load urdf file  """
    workingDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    workingDir += '/Modeles'
    package_dir = workingDir
    urdf_file = workingDir + '/motoman_hc10_support/urdf/hc10dt.urdf'
    robot = RobotWrapper.BuildFromURDF(urdf_file,package_dir,verbose=True)
    return robot

class trajectory:
    def __init__(self):
        """ Init the different values of position, velocity and acceleration"""

        self.dt = dt
        self.X = np.zeros((N,6))
        self.t = np.zeros(N)                            # Initialisation
        self.dotX = np.zeros(self.X.shape)
        self.ddX = np.zeros(self.X.shape)
        self.IDX = robot.model.getFrameId("tool0")

    def lectureCSV(self):
        self.f = open('data.csv','r')                   # read the file
    
        lecteurCSV = csv.reader(self.f)
        self.X = next(lecteurCSV)
        self.dotX = next(lecteurCSV)                    # storage in variables
        self.ddX = next(lecteurCSV)
        self.dt = next(lecteurCSV)
        self.f.close()

    def EcritureFichierCSV(self,N,robot,dt):
        """ Write the different values of position, velocity and acceleration in csv file"""
        self.dt = dt
        self.t = np.zeros(N)
        self.X = np.zeros((N,6))
        self.dotX = np.zeros(self.X.shape)
        self.ddX = np.zeros(self.X.shape)
        self.IDX = robot.model.getFrameId("tool0")
        for i in range(N):
            q,dq,ddq = loiPendule(robot,i*self.dt)              # pendulum law use
            robot.forwardKinematics(q,dq,ddq)
            djv = getdjv(robot,q,dq,ddq)
            pin.updateFramePlacements(robot.model,robot.data)   # update frame placement 
            J = computePlanarJacobian(robot,q,self.IDX)
            
            self.X[i,:] = situationOT(robot.data.oMf[self.IDX])
            self.t[i] = i*self.dt
            self.dotX[i,:] = np.dot(J,dq)
            self.ddX[i,:] = djv + np.dot(J,ddq)
        
        with open('data.csv','w',newline='') as fichiercsv:
            writer=csv.writer(fichiercsv)
            writer.writerow(self.X)
            writer.writerow(self.dotX)                          # write the values
            writer.writerow(self.ddX)
            writer.writerow(self.t)

    def talker_file(self):
        """ Publish the position, velocity and acceleration in topic """
        pub = rospy.Publisher('Topic_file_trajectory', String, queue_size=10) #Topic name to change
        rospy.init_node('talker_file', anonymous=True)
        rate = rospy.Rate(10) # 10hz

        while not rospy.is_shutdown():
            TrajectoryX = "Trajectory for position %s" % rospy.get_time()
            rospy.loginfo(TrajectoryX)
            pub.publish(self.X)

            TrajectorydX = "Trajectory for velocity %s" % rospy.get_time()
            rospy.loginfo(TrajectorydX)
            pub.publish(self.dotX)

            TrajectoryddX = "Trajectory for acceleration %s" % rospy.get_time()
            rospy.loginfo(TrajectoryddX)
            pub.publish(self.ddX)

            rate.sleep()
    
    def Trace(self):
        """ function to trace the values read in csv file"""

        plt.figure()
        plt.plot(self.t,self.X[:,0], 'r--',label="Trajectoire en position X")
        plt.ylabel('Xd en x')
        plt.xlabel('temps en seconde')
        plt.legend()
        plt.figure()
        plt.plot(self.t,self.X[:,1], 'g--',label="Trajectoire en position Y")
        plt.ylabel('Xd en y')
        plt.xlabel('temps en seconde')
        plt.legend()
        plt.figure()
        plt.plot(self.t,self.X[:,2], 'b--',label="Trajectoire en position Z")
        plt.ylabel('dXd en z')
        plt.xlabel('temps en seconde')
        plt.legend()
        plt.figure()
        plt.plot(self.t,self.dotX[:,0], 'r--',label="Vitesse X")
        plt.ylabel('dXd en x')
        plt.xlabel('temps en seconde')
        plt.legend()
        plt.figure()
        plt.plot(self.t,self.dotX[:,1],'g--',label="Vitesse sur Y")
        plt.ylabel('dXd en y')
        plt.xlabel('temps en seconde')
        plt.legend()
        plt.figure()
        plt.plot(self.t,self.dotX[:,2],'b--',label="Vitesse sur Z")
        plt.ylabel('dXd en z')
        plt.xlabel('temps en seconde')
        plt.legend()
        plt.figure()
        plt.plot(self.t,self.ddX[:,0], 'r--',label="Accélération sur X")
        plt.ylabel('ddXd en x')
        plt.xlabel('temps en seconde')
        plt.legend()
        plt.figure()
        plt.plot(self.t,self.ddX[:,1], 'g--',label="Accélération sur Y")
        plt.ylabel('ddXd en y')
        plt.xlabel('temps en seconde')
        plt.legend()
        plt.figure()
        plt.plot(self.t,self.ddX[:,2], 'b--',label="Accélération sur Z")
        plt.ylabel('dXd en z')
        plt.xlabel('temps en seconde')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    N = 5000
    dt = 1e-3
    robot = getRobot()
    traj = trajectory()
    traj.EcritureFichierCSV(N,robot,dt)
    traj.lectureCSV()
    #traj.Trace()
    """try:
        traj.talker_file()
    except rospy.ROSInterruptException:
        pass"""