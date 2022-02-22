from numpy import double, linalg, sign, size, sqrt
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


package_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/Modeles/'
urdf_path = package_path + 'planar_2DOF/URDF/planar_2DOF.urdf'

# ========== Step 1 - load model, create robot model and create robot data

robot = RobotWrapper()
robot.initFromURDF(urdf_path, package_path, verbose=True)
robot.initViewer(loadModel=True)
robot.display(robot.q0)

data = robot.data
model = robot.model
NQ = robot.nq                 # joints angle
NV = robot.nv                 # joints velocity
NJOINT = robot.model.njoints  # number of links
gv = robot.viewer.gui

#sampling time 
Tech=0.01

def trajectory_mode_a2a_sync():
    # data.qlim did not work
    print('i am in' )
    
    mode=0
    while (mode!=1 and mode!=2):
        print('enter 1 for the mode axe to axe, and 2 for the mode syncronized')
        mode=float(input())
        if(mode==1):
            print('enter the total number of joints')
            nbr_joint=int(input())

            print('enter the number of joint you want to move  ')
            joint_i=float(input())

            print('enter lower bound position (q_min)')
            q_min=float(input())

            print('enter uper bound position (q_max)')
            q_max=float(input())
            
            print('enter the MAX velocity of joint')
            V_joint=float(input())

            
            print('enter the acceleration of joint')
            acc_joint=float(input())

            print('enter number of repetition time of motion')
            nbr_rep=int(input())

            Q_plot=calcul_Q_all_variable(nbr_rep,q_min,q_max,V_joint,acc_joint,Tech)
            Q_plot_80=calcul_Q_all_variable(nbr_rep,q_min,q_max,V_joint*0.8,acc_joint,Tech)
            Q_plot_60=calcul_Q_all_variable(nbr_rep,q_min,q_max,V_joint*0.6,acc_joint,Tech)
            Q_plot_40=calcul_Q_all_variable(nbr_rep,q_min,q_max,V_joint*0.4,acc_joint,Tech)
            Q_plot_20=calcul_Q_all_variable(nbr_rep,q_min,q_max,V_joint*0.2,acc_joint,Tech)
            
            # print('here the trajectory of joint',joint_i,'the other joints dont move')
            # plot_Trajectory(Q_plot)
            # plot_Trajectory(Q_plot_80)
            # plot_Trajectory(Q_plot_60)
            # plot_Trajectory(Q_plot_40)
            # plot_Trajectory(Q_plot_20)
            
            Q_total,V_total,A_total=calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot)
            plot_QVA_total(Q_plot,nbr_joint,Q_total,V_total,A_total,'max_')

            Q_total_80,V_total_80,A_total_80=calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot_80)
            plot_QVA_total(Q_plot_80,nbr_joint,Q_total_80,V_total_80,A_total_80,'80_')

            Q_total_60,V_total_60,A_total_60=calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot_60)
            plot_QVA_total(Q_plot_60,nbr_joint,Q_total_60,V_total_60,A_total_60,'60_')

            Q_total_40,V_total_40,A_total_40=calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot_40)
            plot_QVA_total(Q_plot_40,nbr_joint,Q_total_40,V_total_40,A_total_40,'40_')

            Q_total_20,V_total_20,A_total_20=calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot_20)
            plot_QVA_total(Q_plot_20,nbr_joint,Q_total_20,V_total_20,A_total_20,'20_')


        if(mode==2):
            print('mode 2 youpi')
        else:
            print('please re-enter your choice :)')  


    return Q_total,V_total,A_total

def plot_QVA_total(Q_plot,nbr_joint,Q_total,V_total,A_total,name):
    
    plt.figure('Q_total Trajectory')
    for i in range(nbr_joint):
        plt.plot(Q_plot[1],Q_total[i],linewidth=1, label='q'+str(name)+str(i))
    plt.title('q Trajectory')
    plt.xlabel('t')
    plt.ylabel('q')
    plt.legend()
    plt.show()        

    plt.figure('V_total velosity')
    for i in range(nbr_joint):
        plt.plot(Q_plot[1],V_total[i],linewidth=1, label='V'+str(name)+str(i))
    plt.title('V velosity')
    plt.xlabel('t')
    plt.ylabel('V')
    plt.legend()
    plt.show() 

    plt.figure('A_total acceleration')
    for i in range(nbr_joint):
        plt.plot(Q_plot[1],A_total[i],linewidth=1, label='acc'+str(name)+str(i))
    plt.title('acc acceleration')
    plt.xlabel('t')
    plt.ylabel('acc')
    plt.legend()
    plt.show() 

def calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot):
    Q_total=[]
    V_total=[]
    A_total=[]
    tmp=[]
    for i in range(nbr_joint):
                if(i==joint_i):
                    Q_total.append(Q_plot[0])
                else:
                    tmp=[]
                    for i in range(np.array(Q_plot[0]).size):
                        tmp.append(0)
                    Q_total.append(tmp)

    for i in range(nbr_joint):
                if(i==joint_i):
                    V_total.append(Q_plot[2])
                else:
                    tmp=[]
                    for i in range(np.array(Q_plot[2]).size):
                        tmp.append(0)
                    V_total.append(tmp)

    for i in range(nbr_joint):
                if(i==joint_i):
                    A_total.append(Q_plot[3])
                else:
                    tmp=[]
                    for i in range(np.array(Q_plot[3]).size):
                        tmp.append(0)
                    A_total.append(tmp)

    # Q_total.append(V_total)
    # Q_total.append(A_total)

    return Q_total,V_total,A_total

def plot_Trajectory(Q):
    
    q=Q[0],
    time=Q[1]
    v=Q[2]
    a=Q[3]
    print('shape of Q[0]',np.array(Q[0]).shape)
    print('shape of Q[1]',np.array(Q[1]).shape)
    plt.figure('q Trajectory')
    plt.plot(Q[1],Q[0],linewidth=1, label='q')
    plt.title('q Trajectory')
    plt.xlabel('t')
    plt.ylabel('q')
    plt.legend()
    plt.show()

    plt.figure('V velocity calculated via derivation(dq)')
    plt.title('V velocity calculated via derivation(dq)')
    plt.plot(time,v,linewidth=1, label='V')
    plt.xlabel('t sec')
    plt.ylabel('V(m/s)')
    plt.legend()
    plt.show()
    

    plt.figure('acc acceleration calculated via derivation(ddq)')
    plt.plot(time,a,linewidth=1, label='acc')
    plt.xlabel('t sec')
    plt.ylabel('acc (m/s^2)')
    plt.legend()
    plt.show()
    plt.title('acc acceleration calculated via derivation(ddq)')

def calcul_Q_all_variable(nbr_rep,q_min,q_max,V_joint,acc_joint,Tech):
    Q_plot=[]
    Q=[]
    V=[]
    T=[]
    tf=0
    A=[]
    print('avant nbr_rep')
    for i in range(nbr_rep):
        print('avant trajectory')      
        q,t,v,a=trajectory(q_min,q_max,V_joint,acc_joint,Tech)
        print('apres trajectory')  
        Q.extend(q)
        V.extend(v)
        A.extend(a)
        for i in range(np.array(t).size):
            t[i]+=tf

        T.extend(t)       
        tf=T[np.array(T).size-1]

        q1,t1,v1,a1=trajectory(q_max,q_min,V_joint,acc_joint,Tech)
        Q.extend(q1)
        V.extend(v1)
        A.extend(a1)

        for i in range(np.array(t1).size):
            t1[i]+=tf

        T.extend(t1)
        tf=T[np.array(T).size-1]

    Q_plot.append(Q)
    Q_plot.append(T)
    Q_plot.append(V)
    Q_plot.append(A)

    return Q_plot

def trajectory(q_start,q_end,Vmax,acc_max,Tech):

    #function that take as input:
    #                            1-initial position
    #                            2-final position
    #                            3-Max velocity
    #                            4-acceleration
    #                            5- periode of sampling
    #  
    # AND return:
    #            1- trajectory q
    #            2- velocity (dq/dt)
    #            3- acceleration
    #            4- time vector that take the movement to go from qi to qf
     

                                #the max velocity is given by vmax=sqrt(D*ACC)

    D=q_end-q_start             # total distance 
    if(abs(D)>((Vmax*Vmax)/acc_max)):# D>Vmax^2/acc it's condition so the movment can be achivebel(realisable)
        time1=Vmax/acc_max      #time that take the velocity to go from 0 to max
    else:
        print('we need a better value of D')
                                # if D dont respect the condition we need another D 
                                # so another input for the function

    timeEnd=time1+(abs(D)/Vmax)      # time wen the mvt end
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
    v=np.array(v)
    v=abs(v)
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
    # taking data for each joint from the user
    for i in range (Nmbrofjoint):
        print('enter joint',i+1,'start position')
        start=float(input())
        q_start.append(start)
        print('enter joint',i+1,'final position')
        end=float(input())
        q_end.append(end)
        print('enter joint',i+1,'velocity')
        Vm=float(input())
        Vmax.append(Vm)
        print('enter joint',i+1,'acceleration')
        am=float(input())
        acc_max.append(am)
    
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


trajectory_mode_a2a_sync()

'''


# initialisation
q_start=[]
q_end=[]
Vmax=[]
acc_max=[]
Tech=0.0001

# nomber of joint initialisation
nbrjoint=2

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



#calculation of trajectory with several value of velocity
# q_max,time,v_max,a_max=trajectory(10,150,12,2,Tech)
# q1_20,time1,v1_20,a1_20=trajectory(0,100,14,3,Tech)
# q2_40,time2,v2_40,a2_40=trajectory(q_start,q_end,Vmax*0.4,acc_max,Tech)
# q3_60,time3,v3_60,a3_60=trajectory(q_start,q_end,Vmax*0.6,acc_max,Tech)
# q4_80,time4,v4_80,a4_80=trajectory(q_start,q_end,Vmax*0.8,acc_max,Tech)

#Display of trajectory

# plt.figure('q Trajectory')
# plt.plot(time,q_max,  linewidth=1, label='10-150-12-2 position')
# plt.plot(time1,q1_20, linewidth=1, label='0-100-14-2 position')
# # plt.plot(time2,q2_40, linewidth=1, label='q2_40 position')
# plt.plot(time3,q3_60, linewidth=1, label='q3_60 position')
# plt.plot(time4,q4_80, linewidth=1, label='q4_80 position')
# plt.title('q Trajectory')
# plt.xlabel('t')
# plt.ylabel('q')
# plt.legend()
# plt.show()

# #Display of velocity 

# plt.figure('V velocity')
# plt.plot(time,v_max,linewidth=1, label='v_max ')
# plt.plot(time1,v1_20, linewidth=1, label='v1_20 ')
# # plt.plot(time2,v2_40, linewidth=1, label='v2_40 ')
# # plt.plot(time3,v3_60, linewidth=1, label='v3_60 ')
# # plt.plot(time4,v4_80, linewidth=1, label='v4_80 ')
# plt.title('V velocity')
# plt.xlabel('t')
# plt.ylabel('V')
# plt.legend()
# plt.show()

# #Display of acceleration 

# plt.figure('a acceleration')
# plt.plot(time,a_max,linewidth=1, label='a_max ')
# plt.plot(time1,a1_20, linewidth=1, label='a1_20 ')
# plt.plot(time2,a2_40, linewidth=1, label='a2_40 ')
# plt.plot(time3,a3_60, linewidth=1, label='a3_60 ')
# plt.plot(time4,a4_80, linewidth=1, label='a4_80 ')
# plt.title('a acceleration')
# plt.xlabel('t')
# plt.ylabel('a')
# plt.legend()
# plt.show()

'''