# Test affichage robot 3D
# include all librairies 
#from tabulate import tabulate
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import scipy.linalg as sp
import numpy as np
from numpy.core.defchararray import find
from numpy.core.records import array
import pinocchio as pin
from pinocchio import RobotWrapper
from pinocchio.visualize import GepettoVisualizer

# creating robot object
R = RobotWrapper()
package = '/home/fadi/Projet/identification_toolbox/models/others/robots/' 
filename    = package + '/2DOF_description/urdf/2DOF_col_description.urdf'
R.initFromURDF(filename,package)
Njoint=R.model.njoints
nbSample=100
NQ = R.nq

# 3D visualisation
#R.initDisplay(loadModel=True)
#R.display(R.q0)


#generate inertial parameter

phi=[]
for i in range(1,Njoint):
    phi.extend(R.model.inertias[i].toDynamicParameters())

vect_inertie =[]
vect_param =[]

for inertia in R.model.inertias:
    vect_inertie.extend(inertia.toDynamicParameters())
    vect_param.append(inertia.toDynamicParameters())


# creat vect name 
names =[]
for i in range(Njoint):
    names += ['m'+str(i), 'mx'+str(i), 'my'+str(i), 'mz'+str(i), 'Ixx'+str(i), 'Ixy'+str(i), 'Iyy'+str(i), 'Izx'+str(i), 'Izy'+str(i), 'Izz'+str(i)]     

#creat dictionary
dict={names[i]:vect_inertie[i] for i in range(len(names))}
#print(dict)

##3- Generate input/output
## for each joint 100 samples (q,dq,ddq) 

# input
q   = np.random.rand(NQ, nbSample) * np.pi - np.pi/2  #-pi/2< q <pi/2
#print(q.shape)
dq  = np.random.rand(NQ, nbSample) * 10               #0< dq  <10
ddq = np.ones((NQ,nbSample))*2                        #0< ddq <1

#output
tau=[]
for i in range(nbSample):
    tau.extend(pin.rnea(R.model, R.data, q[:, i], dq[:, i], ddq[:, i]))


print('shape de tau: \t',np.array(tau).shape)
#print(tau.shape)


#4- creat IDM by pinocchio
#     for each joint   
#for i in range(NQ):

W=[]
for i in range(nbSample):
    W.extend(pin.computeJointTorqueRegressor(R.model, R.data, q[:, i], dq[:, i], ddq[:, i]))

#5- remove non dynamic effect coluns
#    remove zero value coluns
#    remove the parameters related to zero value coluns
#    at the end we will have a matix W_modifier et Phi_modifier   
threshold=0.000001
W_base = W[:]
W = np.array(W)
print(W.shape)
tmp = []
for i in range(W.shape[1]):
    if (np.dot([W[:, i]], np.transpose([W[:, i]]))[0][0] <= threshold):
        tmp.append(i)

tmp.sort(reverse=True)
#print(tmp)

phi_base = phi[:]
# inertialParameters_base = inertialParameters[:]
names_modified=[]
for i in tmp:
    W = np.delete(W, i, 1)
    phi = np.delete(phi, i, 0)
    names_modified = np.delete(names_modified, i, 0)
print('shape of Wb:\t', np.array(W_base).shape)
print('shape of W:\t', W.shape)

print('shape of phi:\t', np.array(phi).shape)

#6- Q R decomposition 
#       pivoting 
#       we use scipy librairie

(Q,R,P)=sp.qr(W,pivoting=True)

# sort params as decreasing order of diagonal of R
params_rsorted = []
for ind in P:
    params_rsorted.append(phi[ind])

tmp=0
for i in range(np.diag(R).shape[0]):
    if abs(np.diag(R)[i]) < threshold :
        tmp = i

R1 = R[0:tmp, 0:tmp]
Q1 = Q[:, 0:tmp]
R2 = R[0:tmp, tmp:R.shape[1]]
beta = np.round(np.dot(np.linalg.inv(R1), R2), 6)


print('Shape of R1:\t', np.array(R1).shape)
print('Shape of R2:\t', np.array(R2).shape)
print('Shape of Q1:\t', np.array(Q1).shape)



# 8- calculate the Phi_modifier 
#       Phi_modifier=(Q1*R1)^-1*taux
phi = np.round(np.dot(np.linalg.inv(R1), np.dot(Q1.T, tau)), 6)
print('shape of phi \t',phi.shape)
W_b = np.dot(Q1, R1)  # base regressor
print('shape of W_b \t',W_b.shape)


params_idp = params_rsorted[:tmp]
params_rgp = params_rsorted[tmp]

params_base = []
for i in range(tmp):
    if beta[i] == 0:
        params_base.append(params_idp[i])

    else:
        params_base.append(str(params_idp[i]) + ' + '+str(round(float(beta[i]), 6)) + ' * ' + str(params_rgp))

print('base parameters and their identified values: ')
print(params_base)
table = [params_base, phi]
print('finale table shape \t', np.array(table).shape)
print(table)


############################################################

# 9- calculer tau avec phi(paramaetre de base) et W_b le base regressor 

tau_b=np.dot(W_b,phi) 
print('shape of tau_b \t',np.array(tau).shape)
echt=[]
for i in range(nbSample*NQ):
    echt.append(i)
# print(np.array(echt).shape)

# trace le resultat dans un graph
# les deux plot sur la memes figure 
plt.plot(echt,tau_b,color='green',linewidth=2,label="tau_b")#linewidth linestyle
plt.plot(echt,tau,color='blue',linewidth=1,label="tau")
plt.legend()
plt.title("graphe montrant tau et tau_b")
#les deux plot sur deux figures differentes
fig, axs = plt.subplots(2)
fig.suptitle('tau et tau_b separement')
axs[0].plot(echt, tau,color='blue',label="tau")
# plt.legend()
axs[1].plot(echt,tau_b,color='green',label="tau_b")
plt.legend()
# showing results
plt.show()

# 10- calcul phi_etoile moindre carre min abs(tau-phi_etoile*W_b)^2
# on applique le raisonement de moindre carre classique 
# on annulant le gradien de l'erreur avec une Hes>0
w_btw_b =np.linalg.inv(np.dot(W_b.T,W_b))
w_bt_tau=np.dot(W_b.T,tau)
phi_etoile=np.dot(w_btw_b,w_bt_tau)
print('shape of phi_etoile \t',phi_etoile.shape)

# affichage de phi et phi_etoile
echt1=[]
for i in range(phi_etoile.shape[0]):
     echt1.append(i+1)
plt.scatter(echt1,phi,color='green',linewidth=2,label="phi")
plt.scatter(echt1,phi_etoile,color='red',linewidth=1,label="phi etoile")
plt.title("graphe montrant phi et phi etoile")
plt.legend()
plt.show()

# 11- calcul l'err entre tau et tau_b calcule a partir de l'identification
err=[]
for i in range(nbSample*NQ):
    err.append(abs(tau[i]-tau_b[i])*abs(tau[i]-tau_b)[i])


print(np.array(err).shape)
plt.plot(echt,err,label="err")
plt.title("erreur quadratique")
plt.legend()
plt.show()



## step to do

# 1- load model creat robot model and creat robot data that is an attribut for the object robot 
#                (robot model the things that don't change mass length....
#                 robot  data variable featres joint angle joint velosities....)
#                algo: use the model and the data

# 2- generate inertial parameter
#    for all joint 
#    model.inertials[numbre of link(nbr de liaison)].toDynamicparametrs()
#    
#    creat name(vector) foe each parametre concatenat 
#    so in one liste we have to put all the parametres 
#    
#    creat dictionary [name value] key:value    
# 
# 3- Generate input/output 
#    input (joint angle, velositiey,accelration)
#    for each joint 100 samples (q,dq,ddq) randn values or sin,cos d(sin,cos) dd(sin,cos)
#    
#    Generate output (joint torque)(taux)
#    for each joint 100 sample
#    pin.rnea(model,data,q,dq,ddq)
#    and we have to stack the result in a big matrix sample*nbr_of_Joint/colones=parametre*nbr_of_Joint      
# 4- creat IDM by pinocchio
#     for each joint   
#     pin.computeJointTorqueRegressor(model,data,q[i],dq[i],ddq[i])        
#     stack all matrix 
#     at the end we will have a matix W ligne:sample*nbr_of_Joint/colones=parametre*nbr_of_Joint 
#              
# 5- remove non dynamic effect coluns
#    remove zero value coluns
#    remove the parameters related to zero value coluns
#    at the end we will have a matix W_modifier et Phi_modifier        
# 6- Q R decomposition 
#       pivoting 
#       we use scipy librairie
#       scipy.linalg.qr(W_modifier,pivoting=True)                                                   
#                                                              
# 7- calulate the base parameters
#    p_b={p_independant >0(comparaison avec le doagonale de la matrice R dans pivoting) 
#         p_dependant =0}
#   QR=[Q1 Q2][R1 R2] 
#   we calculate the coefficients p_i[i]+beta_j.p_d[k] |beta=R1^-1*R2 
# 
# 8- calculate the Phi_modifier 
#       Phi_modifier=(Q1*R1)^-1*taux
#
# 9- tank you very much 
# 
# 
#                                                            


