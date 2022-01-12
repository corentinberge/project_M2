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

# initialisation
q_start=0
q_end=100
Vmax=14.1
acc_max=2
Tech=0.0001


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

#calculation of trajectory with several value of velocity
q_max,time,v_max,a_max=trajectory(q_start,q_end,Vmax,acc_max,Tech)
q1_20,time1,v1_20,a1_20=trajectory(q_start,q_end,Vmax*0.2,acc_max,Tech)
q2_40,time2,v2_40,a2_40=trajectory(q_start,q_end,Vmax*0.4,acc_max,Tech)
q3_60,time3,v3_60,a3_60=trajectory(q_start,q_end,Vmax*0.6,acc_max,Tech)
q4_80,time4,v4_80,a4_80=trajectory(q_start,q_end,Vmax*0.8,acc_max,Tech)

#Display of trajectory

plt.figure('q Trajectory')
plt.plot(time,q_max,  linewidth=1, label='q_max position')
plt.plot(time1,q1_20, linewidth=1, label='q1_20 position')
plt.plot(time2,q2_40, linewidth=1, label='q2_40 position')
plt.plot(time3,q3_60, linewidth=1, label='q3_60 position')
plt.plot(time4,q4_80, linewidth=1, label='q4_80 position')
plt.title('q Trajectory')
plt.xlabel('t')
plt.ylabel('q')
plt.legend()
plt.show()

#Display of velocity 

plt.figure('V velocity')
plt.plot(time,v_max,linewidth=1, label='v_max ')
plt.plot(time1,v1_20, linewidth=1, label='v1_20 ')
plt.plot(time2,v2_40, linewidth=1, label='v2_40 ')
plt.plot(time3,v3_60, linewidth=1, label='v3_60 ')
plt.plot(time4,v4_80, linewidth=1, label='v4_80 ')
plt.title('V velocity')
plt.xlabel('t')
plt.ylabel('V')
plt.legend()
plt.show()

#Display of acceleration 

plt.figure('a acceleration')
plt.plot(time,a_max,linewidth=1, label='a_max ')
plt.plot(time1,a1_20, linewidth=1, label='a1_20 ')
plt.plot(time2,a2_40, linewidth=1, label='a2_40 ')
plt.plot(time3,a3_60, linewidth=1, label='a3_60 ')
plt.plot(time4,a4_80, linewidth=1, label='a4_80 ')
plt.title('a acceleration')
plt.xlabel('t')
plt.ylabel('a')
plt.legend()
plt.show()

