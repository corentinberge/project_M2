from pyexpat import model
from numpy import double, linalg, sign, sqrt
from numpy.core.fromnumeric import shape
from numpy.lib.nanfunctions import _nanmedian_small
from pinocchio.visualize import GepettoVisualizer
from pinocchio.robot_wrapper import RobotWrapper
import matplotlib.pyplot as plt
import scipy.linalg as sp
import pinocchio as pin
import numpy as np
import os
from typing import Optional
from typing import Optional
import qpsolvers

package_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + '/Yaskawa/Modeles/'
urdf_path = package_path + 'motoman_hc10_support/urdf/hc10_FGV.urdf'


def trajectory(q_start,q_end,Vmax,acc_max,Tech):
    
    #function that take first position ,last position,Max velocity, and acceleration
    # AND return the trajectory of q and the velocity (dq/dt)

                                #the max velocity is given by vmax=sqrt(D*ACC)

    D=q_end-q_start             # total distance 
    if(D>((Vmax*Vmax)/acc_max)):# D>Vmax^2/acc it's condition so the movment can be achivebel(realisable)
        time1=Vmax/acc_max      #time that take the velocity to go from 0 to max
    else:
        print('we need a better value of D')
                                # if D dont respect the condition we need another D 
                                # so another input for the function

    timeEnd=time1+(D/Vmax)      # time wen the mvt end
    print('Vmax \t acc_max \t time1 \t timeEnd')
    print(Vmax,'\t',acc_max,'\t',time1,'\t',timeEnd)
    t=0
    time=[]
    time.append(t)
    q=[]
    v=[]
    a=[]
    #calculation of q in each intervel of time 
    if(t<=time1):
        q.append(q_start+0.5*t*t*acc_max*sign(D))
    elif(t<=(timeEnd-time1)):
        q.append(q_start+(t-(time1/2))*Vmax*sign(D))
    elif(t<=timeEnd):
        q.append(q_end-0.5*(timeEnd-t)*(timeEnd-t)*acc_max*sign(D))
    else:
        print('time out of bound')
    
    while(abs(t-timeEnd)>0.01):
        if(t<=time1):
            q.append(q_start+0.5*t*t*acc_max*sign(D))
        elif(t<=(timeEnd-time1)):
            q.append(q_start+(t-(time1/2))*Vmax*sign(D))
        elif(t<=timeEnd):
            q.append(q_end-0.5*(timeEnd-t)*(timeEnd-t)*acc_max*sign(D))
        else:
            print('time out of bound')
        
        t=t+Tech
        time.append(t)
        
    #calculation of the velocity 
    for i in range(np.array(time).size-1):
        j=i+1
        dv=(q[j]-q[i])/(time[j]-time[i])
        # print('dv=\t',dv)
        v.append(dv)
    v.append(dv)
    print('shape of time',np.array(time).shape)
    print('shape of v',np.array(v).shape)
    
    #calculation of acceleration
    
    for i in range(np.array(v).size-1):
        j=i+1
        da=(v[j]-v[i])/(time[j]-time[i])
        # print('da=\t','dv',(v[j]-v[i]),'/dt',(time[j]-time[i]),'=',da)
        a.append(da)
    a.append(0)
    print('shape of time',np.array(time).shape)
    print('shape of a',np.array(a).shape)

    return q,time,v,a

def Data_Alltrajectory(Nmbrofjoint,Tech):

    # this function take the number of joint and the periode of sampling
    # And return -the first time to go from 0 to max velocity for all joint 
    #            -the final time for all joint to complet the hall trajectory
    #            -the start position q_start
    #            -the final position q_end
    #            -the new joint velocity lamda*Vmax
    #            -the new joint acceleration mu*acc
    #            -the Distance between initial and final position
                  
    q_start=[]
    q_end=[] 
    Vmax=[]
    acc_max=[]
    D=[]
    #-------DATA---
    # taking data for each joint from the urdf
    q_start=[-3.141592653589793, -3.141592653589793, -0.08726646259971647,-3.141592653589793, -3.141592653589793, -3.141592653589793]
    q_end=[3.141592653589793, 3.141592653589793, 6.19591884457987, 3.141592653589793, 3.141592653589793, 3.141592653589793] 
    Vmax=[2.2689280275926285, 2.2689280275926285, 3.141592653589793, 3.141592653589793, 4.363323129985824, 4.363323129985824]
    acc_max=[2,2,2,2,2,2]
    #-------NOW we get it from urdf------------------
    #for i in range (Nmbrofjoint):
        # print('enter joint',i+1,'start position')
        # start=float(input())
        # q_start.append(start)
        # print('enter joint',i+1,'final position')
        # end=float(input())
        # q_end.append(end)
        # print('enter joint',i+1,'velocity')
        # Vm=float(input())
        # Vmax.append(Vm)
        # print('enter joint',i+1,'acceleration')
        # am=float(input())
        # acc_max.append(am)
    #----------------------------------
    #Distance calculation
    for i in range (Nmbrofjoint):
        d=q_end[i]-q_start[i]
        D.append(d)             
    
    # first coefficient calculation lamda_1 and mu_1
    lamda_1=1
    mu_1=1
    lamda_j=[]
    mu_j=[]
    lamda_j.append(lamda_1)
    mu_j.append(mu_1)
    
    for j in range (1,Nmbrofjoint):
        
        lamda=(Vmax[j]*abs(D[0]))/Vmax[0]*abs(D[j])
        mu=(acc_max[j]*abs(D[0]))/acc_max[0]*abs(D[j])
        lamda_j.append(lamda)
        mu_j.append(mu)

    lamda_j=np.array(lamda_j)
    mu_j=np.array(mu_j)
    lamda_1=np.min(lamda_j)
    mu_1=np.min(mu_j)
    

    # calculation of joints coefficients (lamda for each joint and mu for each joint)  
    lamda_joint=[]
    lamda_joint.append(lamda_1)
    for joint in range(1,Nmbrofjoint):
        lamda_joint.append(lamda_1*((Vmax[0]*abs(D[joint]))/(Vmax[joint]*abs(D[0]))))

    mu_joint=[]
    mu_joint.append(mu_1)

    for joint in range(1,Nmbrofjoint):
        mu_joint.append(mu_1*((acc_max[0]*abs(D[joint]))/(acc_max[joint]*abs(D[0]))))

    #Display of coefficient values
    print('LAMDA_J values\n')
    print(lamda_joint)
    print('lamda_1=\t',lamda_1)
    print('MU_J values\n')
    print(mu_joint)
    print('mu_1=\t',mu_1)
    
    # new velocity calculation
    new_Vmax=[]
    NewVmax=lamda_1*Vmax[0]
    new_Vmax.append(NewVmax)
    for j in range(1,Nmbrofjoint):
        new_Vmax.append(lamda_joint[j]*Vmax[j])

    # new acceleration calculation
    new_Amax=[]
    NewAmax=mu_1*acc_max[0]
    new_Amax.append(NewAmax)
    for j in range(1,Nmbrofjoint):
        new_Amax.append(mu_joint[j]*acc_max[j])

    # first time calculation
    time_tau=(lamda_1*Vmax[0])/(mu_1*acc_max[0])

    #final time calculation
    timeEnd=[]
    tf1=((lamda_1*Vmax[0])/(mu_1*acc_max[0]))+(abs(D[0])/(lamda_1*Vmax[0]))
    timeEnd.append(tf1)
    for k in range(1,Nmbrofjoint):     
        if(abs(D[k])>((Vmax[k]*Vmax[k])/acc_max[k])):# D>Vmax^2/acc it's condition so the movment can be achivebel(realisable)
            tf=((lamda_j[k]*Vmax[k])/(mu_j[k]*acc_max[k]))+(D[k]/(lamda_j[k]*Vmax[k]))
            timeEnd.append(tf)                     #time that take the velocity to go from 0 to max
        else:
            print('we need a better value of D for the joint',k+1)
                                # if D dont respect the condition we need another D 
                                # so another input for the function

              # time wen the mvt end
    timeEnd=np.array(timeEnd)
    time_final=np.max(timeEnd)

    return time_tau,time_final,q_start,q_end,new_Vmax,new_Amax,D
    
def PreDefined_trajectory(time1,timeEnd,q_start,q_end, Vmax,acc_max,D,Tech):
    # this function is dedicated for the calculus of a trajectory with predefine data
    # the first time and final time are already calculated 
    # this function return :-position vector q
    #                       -time vector time
    #                       -velocity vector v
    #                       -acceleration vector a

    t=0
    time=[]
    time.append(t)
    q=[]
    v=[]
    a=[]
    #calculation of q in each intervel of time 
    if(t<=time1):
        q.append(q_start+0.5*t*t*acc_max*sign(D))
    elif(t<=(timeEnd-time1)):
        q.append(q_start+(t-(time1/2))*Vmax*sign(D))
    elif(t<=timeEnd):
        q.append(q_end-0.5*(timeEnd-t)*(timeEnd-t)*acc_max*sign(D))
    else:
        print('time out of bound')
    
    while(abs(t-timeEnd)>0.01):
        if(t<=time1):
            q.append(q_start+0.5*t*t*acc_max*sign(D))
        elif(t<=(timeEnd-time1)):
            q.append(q_start+(t-(time1/2))*Vmax*sign(D))
        elif(t<=timeEnd):
            q.append(q_end-0.5*(timeEnd-t)*(timeEnd-t)*acc_max*sign(D))
        else:
            print('time out of bound')
        
        t=t+Tech
        time.append(t)
        
        #calculation of the velocity 
    for i in range(np.array(time).size-1):
        j=i+1
        dv=(q[j]-q[i])/(time[j]-time[i])
        # print('dv=\t',dv)
        v.append(dv)
    v.append(dv)
    print('shape of time',np.array(time).shape)
    print('shape of v',np.array(v).shape)
    
        #calculation of acceleration
    
    for i in range(np.array(v).size-1):
        j=i+1
        da=(v[j]-v[i])/(time[j]-time[i])
        # print('da=\t','dv',(v[j]-v[i]),'/dt',(time[j]-time[i]),'=',da)
        a.append(da)
    a.append(0)
    print('shape of time',np.array(time).shape)
    print('shape of a',np.array(a).shape)

    return q,time,v,a

def PreDefined_velocity_trajectory(time1,timeEnd,Vmax,acc_max,Tech):
        # this function calculate the evolution of velocity  wen time evolute
        # it's based on the law of movement calculation
        # the purpose of this function is to see the theoraticale evolution of velocity 
        # this function return velocity and time vectors
    t=0
    time=[]
    time.append(t)
    q=[]
    v=[]
    a=[]
    #calculation of Velocity in each intervel of time 
    if(t<=time1):
        v.append(acc_max*t)
    elif(t<=(timeEnd-time1)):
        v.append(acc_max*time1)   
    elif(t<=timeEnd):
         v.append(acc_max*(timeEnd-t))
    else:
        print('time out of bound')
    
    while(abs(t-timeEnd)>0.01):
        if(t<=time1):
            v.append(acc_max*t)
        elif(t<=(timeEnd-time1)):
            v.append(acc_max*time1)
        elif(t<=timeEnd):
             v.append(acc_max*(timeEnd-t))
        else:
            print('time out of bound')
        
        t=t+Tech
        time.append(t)
    return v,time


#======MAIN YASKAWA=======
# initialisation
q_start=[]
q_end=[]
Vmax=[]
acc_max=[]
Tech=0.0001

# nomber of joint initialisation
nbrjoint=6
#Data calculation for all joint 
time_tau,time_final,q_start,q_end,Vmax,acc_max,D=Data_Alltrajectory(nbrjoint,Tech)

# calulation and display of theoratical velocity
plt.figure('V velocity theoratical')
for i in range(nbrjoint):
    v1,time1=PreDefined_velocity_trajectory(time_tau,time_final,Vmax[i],acc_max[i],Tech)
    plt.plot(time1,v1,linewidth=1, label='V'+str(i))
plt.title('V velocity th')
plt.xlabel('t')
plt.ylabel('V')
plt.legend()
plt.show()


# calculation and display for all trajectorys
plt.figure('q Trajectory')
for i in range(nbrjoint):
    q1,time1,v1,a1=PreDefined_trajectory(time_tau,time_final,q_start[i],q_end[i], Vmax[i],acc_max[i],D[i],Tech)
    plt.plot(time1,q1,linewidth=1, label='q'+str(i))

plt.title('q Trajectory')
plt.xlabel('t')
plt.ylabel('q')
plt.legend()
plt.show()



#calculation and Display of all velocity (dq)

plt.figure('V velocity calculated via derivation(dq)')
for i in range(nbrjoint):
    q1,time1,v1,a1=PreDefined_trajectory(time_tau,time_final,q_start[i],q_end[i], Vmax[i],acc_max[i],D[i],Tech)
    plt.plot(time1,v1,linewidth=1, label='V'+str(i))
plt.xlabel('t')
plt.ylabel('V')
plt.legend()
plt.show()
plt.title('V velocity calculated via derivation(dq)')



