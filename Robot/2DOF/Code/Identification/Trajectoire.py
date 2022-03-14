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
    #this function dont take an input data  but ask the user to enter his own data: 
    # it returns two mode of trajectorys:
    #   mode 1: axe to axe
    #   1-it return a specifique trajectory for a chosen joint
    #   and for other joints trajectory with zero value
    #   so other joints dont move
    #   2-it return the fricion force of coulomb
    
    # data.qlim did not work
    Q_total=[]
    V_total=[]
    A_total=[]
    Q_total_per_joint=[]
    mode=0
    while (mode!=1 and mode!=2):
        
        # entering data from user 

        print('enter 1 for the mode axe to axe, and 2 for the mode syncronized')
        mode=float(input())
        if(mode==1):
            print('enter the total number of joints')
            nbr_joint=int(input())

            print('enter the number of joint you want to move (counting start from 0) ')
            joint_i=float(input())

            print('enter lower bound position (q_min)')
            q_min=float(input())

            print('enter uper bound position (q_max)')
            q_max=float(input())
            
            print('enter the MAX velocity of joint')
            V_joint=float(input())

            print('enter the desire  prcentatge of the MAX velocity of joint')
            pourcentage=float(input())
            
            print('enter the acceleration of joint')
            acc_joint=float(input())

            print('enter number of repetition time of motion')
            nbr_rep=int(input())
            
            # calculs of position velosity acceleration for the chosen joint with variation 
            # of the Vmax 80% 60% 40% 20%

            Q_plot=calcul_Q_all_variable_a2a(nbr_rep,q_min,q_max,V_joint,acc_joint,Tech)
            Q_plot_80=calcul_Q_all_variable_a2a(nbr_rep,q_min,q_max,V_joint*0.8,acc_joint,Tech)
            Q_plot_60=calcul_Q_all_variable_a2a(nbr_rep,q_min,q_max,V_joint*0.6,acc_joint,Tech)
            Q_plot_40=calcul_Q_all_variable_a2a(nbr_rep,q_min,q_max,V_joint*0.4,acc_joint,Tech)
            Q_plot_20=calcul_Q_all_variable_a2a(nbr_rep,q_min,q_max,V_joint*0.2,acc_joint,Tech)

            Q_plot_pourcentage=calcul_Q_all_variable_a2a(nbr_rep,q_min,q_max,V_joint*pourcentage,acc_joint,Tech)
            
            # position vector Q_plot[0]
            # time vector Q_plot[1]
            # velosity vector Q_plot[2]
            # acc vector Q_plot[3]
            time=Q_plot[1]

            # plot the data of the chosen joint

            # print('here the trajectory of joint',joint_i,'the other joints dont move')
            # plot_Trajectory(Q_plot)
            # plot_Trajectory(Q_plot_80)
            # plot_Trajectory(Q_plot_60)
            # plot_Trajectory(Q_plot_40)
            # plot_Trajectory(Q_plot_20)
            
            # calculs of position velosity acceleration for all joint joint with variation 
            # of the Vmax

            Q_total,V_total,A_total=calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot)
            plot_QVA_total(Q_plot[1],nbr_joint,Q_total,V_total,A_total,'max_')

            # Q_total_80,V_total_80,A_total_80=calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot_80)
            # plot_QVA_total(Q_plot_80[1],nbr_joint,Q_total_80,V_total_80,A_total_80,'80_')

            # Q_total_60,V_total_60,A_total_60=calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot_60)
            # plot_QVA_total(Q_plot_60[1],nbr_joint,Q_total_60,V_total_60,A_total_60,'60_')

            # Q_total_40,V_total_40,A_total_40=calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot_40)
            # plot_QVA_total(Q_plot_40[1],nbr_joint,Q_total_40,V_total_40,A_total_40,'40_')

            # Q_total_20,V_total_20,A_total_20=calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot_20)
            # plot_QVA_total(Q_plot_20[1],nbr_joint,Q_total_20,V_total_20,A_total_20,'20_')

            # Q_total_pourcentage,V_total_pourcentage,A_total_pourcentage=calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot_pourcentage)
            # plot_QVA_total(Q_plot_pourcentage[1],nbr_joint,Q_total_pourcentage,V_total_pourcentage,A_total_pourcentage,'%_')

            tau,w=Generate_Torque_Regression_matrix(nbr_joint,Q_total,V_total,A_total)
            phi_etoile=estimation_with_qp_solver(w,tau)
            Generate_text_data_file(Q_total,V_total,A_total,tau)
            
            force=force_coulomb(phi_etoile[21],V_total,nbr_joint)

            plt.figure('force friction')
            for i in range(nbr_joint):
              plt.plot(V_total[i],force[i],linewidth=1, label='fric'+str(i))
            plt.title('friction force')
            plt.xlabel('v')
            plt.ylabel('fric')
            plt.legend()
            plt.show() 

        if(mode==2):

            print('enter the total number of joints')
            nbr_joint=int(input())
            print('enter number of repetition time of motion')
            nbr_rep=int(input())

            time_tau,time_final,q_start,q_end,Vmax,acc_max,D=Data_Alltrajectory(nbr_joint,Tech)
            print('\n')
            print('the new values of velosity and acc after synchronisation')
            for i in range(nbr_joint):
                print('V_joint',i+1,'\t',Vmax[i],'m/s')
                print('acc_joint',i+1,'\t',acc_max[i],'m/s^2')

            for i in range(nbr_joint):

                # q,time,v,a=PreDefined_trajectory(time_tau,time_final,q_start[i],q_end[i], Vmax[i],acc_max[i],D[i],Tech)
                Q_total_per_joint=calcul_Q_all_variable_sync(nbr_rep,time_tau,time_final,q_start[i],q_end[i],Vmax[i],acc_max[i],D[i],Tech)

                Q_total.append(Q_total_per_joint[0])
                V_total.append(Q_total_per_joint[2])
                A_total.append(Q_total_per_joint[3])
                time=Q_total_per_joint[1]

            plot_QVA_total(time,nbr_joint,Q_total,V_total,A_total,'sync')

            tau,w=Generate_Torque_Regression_matrix(nbr_joint,Q_total,V_total,A_total)
            Generate_text_data_file(Q_total,V_total,A_total,tau)


            phi_etoile=estimation_with_qp_solver(w,tau)
            force=force_coulomb(phi_etoile[21],V_total,nbr_joint)
            plt.figure('force friction')
            for i in range(nbr_joint):
              plt.plot(V_total[i],force[i],linewidth=1, label='fric'+str(i))
            plt.title('friction force')
            plt.xlabel('v')
            plt.ylabel('fric')
            plt.legend()
            plt.show() 

            
        else:
            print('please re-enter your choice :)')  


    return Q_total,V_total,A_total,time,force

def Generate_text_data_file(Q_total,V_total,A_total,tau):
# this function take in input q dq ddq tau for all the joint 
# and write all the data in a file .txt
    package_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
    file_path = package_path + '/Code/Identification/2dof_data_LC_V3_syncronized.txt'

    # f = open('/home/fadi/projet_cobot_master2/project_M2/Robot/2DOF/Code/Identification/2dof_data_LC.txt','w')
    f = open(file_path,'w')
    # tau=[]
    # q_pin = np.random.rand(NQ, nbSamples) * np.pi - np.pi/2  # -pi/2 < q < pi/2
    # dq_pin = np.random.rand(NQ, nbSamples) * 10              # 0 < dq  < 10
    # ddq_pin = np.random.rand(NQ, nbSamples) * 2               # 0 < dq  < 2

    nbSamples=np.array(Q_total[0]).size
    q_pin=np.array(Q_total)
    dq_pin=np.array(V_total)
    ddq_pin=np.array(A_total)
    tau=np.array(tau)
    print('shape of Q ',q_pin.shape)
    print('shape of tau ',tau.shape)

    # tau,w = Generate_Torque_Regression_matrix(NQ,q_pin,dq_pin,ddq_pin)
    i=0
    j=1
    tau1=[]
    tau2=[]
    print('shape of tau',np.array(tau).shape)
    while i<=(tau.size-2):
        tau1.append(tau[i])
        i+=2
    tau1=np.array(tau1)
    print('shape of tau 1',tau1.shape)
    while j<=(tau.size-1):
        tau2.append(tau[j])
        j+=2
    tau2=np.array(tau2)
    print('shape of tau 2', tau2.shape)

    line=[str('q1'),'\t',str('q2'),'\t',str('dq1'),'\t',str('dq2'),'\t',str('ddq1'),
                '\t',str('ddq2'),'\t',str('tau1'),'\t',str('tau2')]
    f.writelines(line)
    f.write('\n')
        

    for i in range(nbSamples):
        line=[str(q_pin[0][i]),'\t',str(q_pin[1][i]),'\t',str(dq_pin[0][i]),'\t',str(dq_pin[1][i]),'\t',str(ddq_pin[0][i]),
                '\t',str(ddq_pin[1][i]),'\t',str(tau1[i]),'\t',str(tau2[i])]
        f.writelines(line)
        f.write('\n')
        
    f.close()

def plot_QVA_total(time,nbr_joint,Q_total,V_total,A_total,name):
    # # this function take in input: position of the joint qi
    #                                velosity of the joint dqi
    #                                acceleration of the joint ddqi
    # the function dont return any thing 
    # it plot: the trajectorys of all the joints
            #  the velositys of all joints
            #  the acceleration of all joints
    samples=[]
    for i in range(np.array(Q_total[0]).size):
        samples.append(i)
    
    time=samples

    plt.figure('Q_total Trajectory')
    for i in range(nbr_joint):
        plt.plot(time,Q_total[i],linewidth=1, label='q'+str(name)+str(i))
    plt.title('q Trajectory')
    plt.xlabel('t')
    plt.ylabel('q')
    plt.legend()
    plt.show()        

    plt.figure('V_total velosity')
    for i in range(nbr_joint):
        plt.plot(time,abs(np.array(V_total[i])),linewidth=1, label='V'+str(name)+str(i))
    plt.title('V velosity')
    plt.xlabel('t sec')
    plt.ylabel('V m/sec')
    plt.legend()
    plt.show() 

    plt.figure('A_total acceleration')
    for i in range(nbr_joint):
        plt.plot(time,A_total[i],linewidth=1, label='acc'+str(name)+str(i))
    plt.title('acc acceleration')
    plt.xlabel('t')
    plt.ylabel('acc')
    plt.legend()
    plt.show() 

def calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot):
    # this function take in input : number of joints 
    #                               a specifique joint joint_i
    #                               and Q_plot the data of joint_i
    # it return the data of al joint in mode axe to axe so the non chosen joint will have:
    #                                                                     q=0,dq=0,ddq=0  
    #                               
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
    # this function plot a pre-calculated trajectory 
    # so it plot q dq ddq in 3 different figure 
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
    plt.plot(time,abs(np.array(v)),linewidth=1, label='V')
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

def calcul_Q_all_variable_sync(nbr_rep,time1,timeEnd,q_min,q_max,V_joint,acc_joint,D,Tech):
    #this function take in input :1- the number of repetition 
    #                             2- the data of each joint motion(qstart qend V acc )
    # and return a matrix that combine:1- position vector after  repetion 
    #                                  2- velosity vector after  repetion 
    #                                  3- acc vector after  repetion 
    #                                  4- time vector with after repetion     

    Q_plot=[]
    Q=[]
    V=[]
    T=[]
    tf=0
    A=[]
    # print('avant nbr_rep')
    for i in range(nbr_rep):
        # print('avant trajectory')      
        q,t,v,a=PreDefined_trajectory(time1,timeEnd,q_min,q_max,V_joint,acc_joint,D,Tech)
        # print('apres trajectory')  
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

def calcul_Q_all_variable_a2a(nbr_rep,q_min,q_max,V_joint,acc_joint,Tech):
     #this function take in input :1- the number of repetition 
    #                             2- the data of the chosen joint motion(qstart qend V acc )
    # and return the data of the chosen joint in a matrix that combine:
    #                                                       1- position vector after  repetion 
    #                                                       2- velosity vector after  repetion 
    #                                                       3- acc vector after  repetion 
    #                                                       4- time vector with after repetion     

    Q_plot=[]
    Q=[]
    V=[]
    T=[]
    tf=0
    A=[]
    # print('avant nbr_rep')
    for i in range(nbr_rep):
        # print('avant trajectory')      
        q,t,v,a=trajectory(q_min,q_max,V_joint,acc_joint,Tech)
        # print('apres trajectory')  
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
    vreturn=v
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

    return q,time,vreturn,a

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
    v=np.array(v)
    vreturn=v
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

    return q,time,vreturn,a

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

def Generate_Torque_Regression_matrix(nbr_joint,Q_total,V_total,A_total):

    #this function take as input number of joints and the data of each joint (q dq ddq)
    #it return the torque of each joint and the regression matrix

    # Generate ouput with pin
    nbSamples=np.array(Q_total[0]).size
    tau=[]
    q=np.array(Q_total)
    dq=np.array(V_total)
    ddq=np.array(A_total)
    print('shape of q test1 \t','\t',q.shape)

    for i in range(nbSamples):
        tau.extend(pin.rnea(model, data, q[:, i], dq[:, i], ddq[:, i]))
    # print('Shape of tau_pin:\t', np.array(tau).shape)
    tau=np.array(tau)
    tau=np.double(tau)
    print ('shape of tau in the function generate output',tau.shape)

    # # ========== Step 4 - Create IDM with pinocchio (regression matrix)
    w = []  # Regression vector
        ## w pour I/O generer par pinocchio
    for i in range(nbSamples):
        w.extend(pin.computeJointTorqueRegressor(model, data, q[:, i], dq[:, i], ddq[:, i]))
    w=np.array(w)

    ## modification of W so it contain dq et singe(dq) for the friction param Fv et Fs
    dq_stack=[]
    for i in range(nbr_joint):
        dq_stack.extend(dq[i])
    
    dq_stack=np.array([dq_stack])
    dq_stack=dq_stack.T

    # calculs of  signe(dq)
    dq_sign=np.sign(dq_stack)

    # modification of w
    w=np.concatenate([w,dq_stack], axis=1)
    w=np.concatenate([w,dq_sign], axis=1)

    return tau,w

def estimation_with_qp_solver(w,tau):
    # this function take in input the regression matrix and the torque vectore of the joint
    # it return the estimeted parameters 
    P = np.dot(w.transpose(),w)
    q = -np.dot(tau.transpose(),w)

    # test if P is positive-definite if not then p=spd (spd=symmetric positive semidefinite)
    P=nearestPD(P)

    #constraints
    #Any constraints that are >= must be multiplied by -1 to become a <=.
    G=([-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],   
    [0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])

    h=[0,-0.05,0.3,-0.05,0.3,-0.05,0.3,0,-0.05,0.3,-0.05,0.3,-0.05,0.3]

    # converting to double
    G=np.array(G)
    h=np.array(h)
    G=np.double(G)
    h=np.double(h)

    # phi_etoile=qpsolvers.solve_ls(P,q,None,None)
    phi_etoile=qpsolvers.solve_qp(
            P,
            q,
            G,#G Linear inequality matrix.
            h,#Linear inequality vector.
            A=None,
            b=None,
            lb=None,
            ub=None,
            solver="quadprog",
            initvals=None,
            sym_proj=True
            )

    tau_estime=np.dot(w,phi_etoile)
    samples = []
    for i in range(np.array(tau).size):
        samples.append(i)

    # plt.figure('torque et torque estime')
    # plt.plot(samples, tau, 'g', linewidth=1, label='tau')
    # plt.plot(samples,tau_estime, 'b:', linewidth=0.4, label='tau estime')
    # # plt.plot(samples, tau_estime1, 'r', linewidth=1, label='tau estime 1')
    # plt.title('tau and tau_estime')
    # plt.xlabel('2000 Samples')
    # plt.ylabel('parametres')
    # plt.legend()
    # plt.show()

    err = []
    for i in range(np.array(tau).size):
        err.append(abs(tau[i] - tau_estime[i]) * abs(tau[i] - tau_estime[i]))
    plt.plot(samples, err, linewidth=2, label="err")
    plt.title("erreur quadratique")
    plt.legend()
    plt.show()



    return phi_etoile

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd 

    spd=symmetric positive semidefinite

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)#provides another way to factorize a matrix, into singular vectors and singular values

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))#Return the distance between norm(A) and the nearest adjacent number
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        #Cholesky's method serves a test of positive definiteness
        # The Cholesky decomposition (or the Cholesky factorization) 
        # is the factorization of a matrix A into the product of a lower triangular matrix L and its transpose. 
        # We can rewrite this decomposition in mathematical notation as: A = L·LT .
        return True
    except np.linalg.LinAlgError:
        return False

def force_coulomb(FS,V_total,nbr_joint):
    # this function take as input : number of joints
    #                               Velosity of joints
    #                               the friction parametres 
    # it return: the coulombs force vector for each joint 
    Force=[]
    V_total=np.array(V_total)
    for j in range(nbr_joint):
        F=[]
        for i in range(V_total[j].size):
            f=FS*np.sign(V_total[j,i])
            # f=FS*(V_total[j,i])
            f=double(f)
            F.extend([f])

        Force.append(F)   

    print(np.array(Force).shape)

    
    return(Force)

def generation_palier_vitesse_calcul_oneJoint(nbr_rep,prct,q_start,q_end,Vmax,acc_max,Tech):
    #this function take in input :1- the number of repetition 
    #                             2- the data of the chosen joint motion(qstart qend V acc)
    
    # and return the data of the chosen joint in a matrix that combine:
    #                                                       1- position vector after  repetion 
    #                                                       2- velosity vector after  repetion 
    #                                                       3- acc vector after  repetion 
    #                                                       4- time vector with after repetion     
    # print('entrer votre pourcentage de augmenter la vitesse')
    # prct=float(input())
    # prct=1
    vitesse=prct*Vmax
    prct_var=prct
    i=0
    loop=0 # pour un test dans ma tete
    Q_all=[]
    Q=[]
    V=[]
    T=[]
    tf=0
    A=[]
    while(vitesse<=Vmax):
        Q_palier=calcul_Q_all_variable_a2a(nbr_rep,q_start,q_end,vitesse,acc_max,Tech)
        #Q_all.append(Q)
        prct_var+=prct
        vitesse=prct_var*Vmax
        loop+=1
        # print('avant trajectory')      
        q=Q_palier[0]
        t=Q_palier[1]
        v=Q_palier[2]
        a=Q_palier[3]
        # print('apres trajectory')  
        Q.extend(q)
        V.extend(v)
        A.extend(a)
        for i in range(np.array(t).size):
            t[i]+=tf

        T.extend(t)       
        tf=T[np.array(T).size-1]

    Q_all.append(Q)
    Q_all.append(T)
    Q_all.append(V)
    Q_all.append(A)

    return Q_all

def generation_palier_vitesse_calcul_allJoint(nbr_rep,prct,nbr_joint,q_start,q_end,Vmax,acc_max,Tech):
    
    tf=0
    T=[]
    Q_palier_V_Joint=[]
    Q_total_one_joint=[]
    V_total_one_joint=[]
    A_total_one_joint=[]
    Q_total_All_Joint=[[],[],[],[],[],[]]
    V_total_All_Joint=[[],[],[],[],[],[]]
    A_total_All_Joint=[[],[],[],[],[],[]]

    # print('Vmax[0]',Vmax[0])
    
    NbSample_interpolate=100
    # j=1
    for i in range(nbr_joint):

        Q_palier_V_Joint=generation_palier_vitesse_calcul_oneJoint(nbr_rep,prct,q_start[i],q_end[i],Vmax[i],acc_max[i],Tech)
        Q_total_one_joint,V_total_one_joint,A_total_one_joint=calcul_QVA_joints_total(nbr_joint,i,Q_palier_V_Joint)
        Q_total_All_Joint=np.concatenate([Q_total_All_Joint,Q_total_one_joint], axis=1)
        V_total_All_Joint=np.concatenate([V_total_All_Joint,V_total_one_joint], axis=1)
        A_total_All_Joint=np.concatenate([A_total_All_Joint,A_total_one_joint], axis=1)


        t=Q_palier_V_Joint[1]
        times=t
        for j in range(np.array(t).size):
            t[j]+=tf
        T.extend(t)       
        tf=T[np.array(T).size-1]

        
        q1,dq1,ddq1,times=generateQuinticPolyTraj_version_GF(Jci_aprBute[i],Jcf_aprBute[i],Vmax[i],Tech)
        # q1,dq1,ddq1=Bang_Bang_acceleration_profile(Jci2[i][0],Jcf2[i][0],Jci2[i][1],Jci2[i][2],Tech)
        Q_inter1_one_joint,V_inter1_one_joint,A_inter1_one_joint=calcul_QVA_joints_total(nbr_joint,i,[q1,times,dq1,ddq1])
        Q_total_All_Joint=np.concatenate([Q_total_All_Joint,Q_inter1_one_joint], axis=1)
        V_total_All_Joint=np.concatenate([V_total_All_Joint,V_inter1_one_joint], axis=1)
        A_total_All_Joint=np.concatenate([A_total_All_Joint,A_inter1_one_joint], axis=1)
        t=times

        for k in range(np.array(t).size):
            t[k]+=tf
        T.extend(t)       
        tf=T[np.array(T).size-1]

        if(i<nbr_joint-1):
            q,dq,ddq,times=generateQuinticPolyTraj_version_GF(Jci_avBute[i+1],Jcf_avBute[i+1],Vmax[i+1],Tech)
            # q,dq,ddq=Bang_Bang_acceleration_profile(Jci1[i+1][0],Jcf1[i+1][0],Jci1[i+1][1],Jci1[i+1][2],Tech)
            Q_inter2_one_joint,V_inter2_one_joint,A_inter2_one_joint=calcul_QVA_joints_total(nbr_joint,i+1,[q,times,dq,ddq])
            Q_total_All_Joint=np.concatenate([Q_total_All_Joint,Q_inter2_one_joint], axis=1)
            V_total_All_Joint=np.concatenate([V_total_All_Joint,V_inter2_one_joint], axis=1)
            A_total_All_Joint=np.concatenate([A_total_All_Joint,A_inter2_one_joint], axis=1)
            # j=j+1
        t=times

        for l in range(np.array(t).size):
            t[l]+=tf
        T.extend(t)       
        tf=T[np.array(T).size-1]
        # plot_QVA_total(times,nbr_joint,Q_inter_one_joint,V_inter_one_joint,A_inter_one_joint,'max_')
        
       
        


            
    print('shape of Time vect T',np.array(T).shape)
    print('Q_total_All_Joint',np.array(Q_total_All_Joint).shape)
    return T,Q_total_All_Joint,V_total_All_Joint,A_total_All_Joint
    
def generateQuinticPolyTraj_version_GF(Jc0,Jcf,vmax,Tech):

    # q=np.zeros(NbSample_interpolate)
    # dq=np.zeros(NbSample_interpolate)
    # ddq=np.zeros(NbSample_interpolate)
    # q=[]
    # dq=[]
    # ddq=[]
    # vmax=vmax*0.75
    # amax=30

    # D=(Jcf[0]-Jc0[0])
    # # vect=[(15*D)/(8*vmax),sqrt((10*D)/(1.73*amax))]
    # tf=(15*abs(D))/(8*vmax)#np.max(np.array(vect))
    
    # a=np.zeros(6)
    # a[0]=Jc0[0]
    # a[1]=Jc0[1]
    # a[2]=Jc0[2]/2
    # a[3]=( 20*Jcf[0]-20*Jc0[0] -(8*Jcf[1]+12*Jc0[1])*tf -(3*Jc0[2]-Jcf[2])*tf**2 )/(2*tf**3)
    # a[4]=( 30*Jc0[0]-30*Jcf[0] +(14*Jcf[1]+16*Jc0[1])*tf +(3*Jc0[2]-2*Jcf[2])*tf**2 )/(2*tf**4)
    # a[5]=( 12*Jcf[0]-12*Jc0[0] -(6*Jcf[1]+6*Jc0[1])*tf -(Jc0[2]-Jcf[2])*tf**2 )/(2*tf**5)

    # t=0
    T=[]
      
    # while(t<tf):
    #     T.append(t)
    #     q_=a[0]+a[1]*t +a[2]*t**2 +a[3]*t**3   +a[4]*t**4     +a[5]*t**5
    #     q.append(q_)
    #     dq_=    a[1]   +2*a[2]*t  +3*a[3]*t**2 +4*a[4]*t**3    +5*a[5]*t**4
    #     dq.append(dq_)
    #     ddq_= +2*a[2]    +6*a[3]*t    +12*a[4]*t**2    +20*a[5]*t**3   
    #     ddq.append(ddq_)
    #     t=t+Tech

    # print('T shape',np.array(T).shape)
    # print('Q shape',np.array(q).shape)

    
    tf=15*np.abs(Jcf[0]-Jc0[0])/(8*vmax)
    NbSample_interpolate=int (tf/Tech) +1

    q=np.zeros(NbSample_interpolate)
    dq=np.zeros(NbSample_interpolate)
    ddq=np.zeros(NbSample_interpolate)

    A=np.array([[1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0 ],
            [1, tf, tf**2, tf**3, tf**4, tf**5],
            [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
            [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]])



    a=np.matmul(np.linalg.inv(A),np.hstack((Jc0,Jcf)))


    t=0
    for i in range(NbSample_interpolate):
    
        q[i]=a[0]+a[1]*t +a[2]*t**2 +a[3]*t**3   +a[4]*t**4      +a[5]*t**5
        dq[i]=    a[1]   +2*a[2]*t  +3*a[3]*t**2 +4*a[4]*t**3    +5*a[5]*t**4
        ddq[i]=          +2*a[2]    +6*a[3]*t    +12*a[4]*t**2    +20*a[5]*t**3      
        t=t+Tech

    return q, dq ,ddq,T

def trajectory_axe2axe_palier_de_vitesse(nbr_joint,nbr_rep,joint_i,prct,q_start,q_end,Vmax,acc_max):
    
    Q_total_All_Joint=[[],[]]
    V_total_All_Joint=[[],[]]
    A_total_All_Joint=[[],[]]
    Q_plot=[]
    q=[]
    dq=[]
    ddq=[]
    times=[]
    
    Jcf_Home=np.array([0,0,0])
    Jci_avBute=np.array([-3.14,0,6])
    
    q,dq,ddq,times=generateQuinticPolyTraj_version_GF(Jcf_Home,Jci_avBute,Vmax,Tech)
    
    Q_plot.append(q)
    Q_plot.append(times)
    Q_plot.append(dq)
    Q_plot.append(ddq)
    
    Q_inter_Home_joint,V_inter_Home_joint,A_inter_Home_joint=calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot)

    Q_total_All_Joint=np.concatenate([Q_total_All_Joint,Q_inter_Home_joint], axis=1)
    V_total_All_Joint=np.concatenate([V_total_All_Joint,V_inter_Home_joint], axis=1)
    A_total_All_Joint=np.concatenate([A_total_All_Joint,A_inter_Home_joint], axis=1)

    Q_plot=generation_palier_vitesse_calcul_oneJoint(nbr_rep,prct,q_start,q_end,Vmax,acc_max,Tech)
    Q_total,V_total,A_total=calcul_QVA_joints_total(nbr_joint,joint_i,Q_plot)

    Q_total_All_Joint=np.concatenate([Q_total_All_Joint,Q_total], axis=1)
    V_total_All_Joint=np.concatenate([V_total_All_Joint,V_total], axis=1)
    A_total_All_Joint=np.concatenate([A_total_All_Joint,A_total], axis=1)

    return Q_total_All_Joint,V_total_All_Joint,A_total_All_Joint



if __name__=="__main__":
    
    
    Q_total=[]
    V_total=[]
    A_total=[]
    nbr_joint=2
    nbr_rep=10
    joint_i=0
    prct=0.1
    q_start=-3.14
    q_end=3.14
    Vmax=2.27
    acc_max=6

    trajectory_mode_a2a_sync() 

    # Q_total,V_total,A_total=trajectory_axe2axe_palier_de_vitesse(nbr_joint,nbr_rep,joint_i,prct,q_start,q_end,Vmax,acc_max)
    # plot_QVA_total(Q_total[1],nbr_joint,Q_total,V_total,A_total,'max_')

    # tau,w=Generate_Torque_Regression_matrix(nbr_joint,Q_total,V_total,A_total)
    # phi_etoile=estimation_with_qp_solver(w,tau)
    
    # Generate_text_data_file(Q_total,V_total,A_total,tau)


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
