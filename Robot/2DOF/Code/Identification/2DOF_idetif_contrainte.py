from numpy import double, linalg
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

# from typing import Matrix, Vector

from typing import Optional
import qpsolvers

# # urdf directory path
# package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# urdf_path = package_path + '/robots/urdf/planar_2DOF.urdf'

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

# ========== Step 2 - generate inertial parameters for all links (excepted the base link)

names = []
for i in range(1, NJOINT):
    names += ['m'+str(i), 'mx'+str(i), 'my'+str(i), 'mz'+str(i), 'Ixx'+str(i),
              'Ixy'+str(i), 'Iyy'+str(i), 'Izx'+str(i), 'Izy'+str(i), 'Izz'+str(i)]

phi = []
for i in range(1, NJOINT):
    phi.extend(model.inertias[i].toDynamicParameters())

print('shape of phi:\t', np.array(phi).shape)

# ========== Step 3 - Generate input and output - 1000 samples than 

q1=[]
q2=[]
dq1=[]
dq2=[]
ddq1=[]
ddq2=[]
tau1=[]
tau2=[]
q=[]
dq=[]
ddq=[]
tau=[]

# ouvrir le fichier data_2dof
f = open('/home/fadi/projet_cobot_master2/project_M2/Robot/2DOF/Code/Identification/data_2dof.txt','r')
# f = open('data_2dof.txt','r')
# getting data from file
for line in f:
    data_split = line.strip().split('\t')
    q1.append(data_split[0])
    q2.append(data_split[1])
    dq1.append(data_split[2])
    dq2.append(data_split[3])
    ddq1.append(data_split[4])
    ddq2.append(data_split[5])
    tau1.append(data_split[6])
    tau2.append(data_split[7])
f.close()
## the data file was modified so we eliminate the first line who dont contain numbers
##put all the q in one array and convert to double
q.append(q1)
q.append(q2)
q=np.array(q)
q=np.double(q)
print('shape of q ',q.shape)

##put all the dq in one array and convert to double
dq.append(dq1)
dq.append(dq2)
dq=np.array(dq)
dq=np.double(dq)
print('shape of dq',dq.shape)

##put all the ddq in one array and convert to double
ddq.append(ddq1)
ddq.append(ddq2)
ddq=np.array(ddq)
ddq=np.double(ddq)
print('shape of ddq',ddq.shape)


##put all the torque values in one array and convert to double
tau.extend(tau1)
tau.extend(tau2)
tau=np.array(tau)
tau=np.double(tau)
print('shape of tau',tau.shape)

nbSamples = 1000  # number of samples

# # Generate inputs with pin (pin as pinocchio)
q_pin = np.random.rand(NQ, nbSamples) * np.pi - np.pi/2  # -pi/2 < q < pi/2
dq_pin = np.random.rand(NQ, nbSamples) * 10              # 0 < dq  < 10
ddq_pin = np.random.rand(NQ, nbSamples) * 2               # 0 < dq  < 2
tau_pin = []
# Generate ouput with pin
for i in range(nbSamples):
    tau_pin.extend(pin.rnea(model, data, q_pin[:, i], dq_pin[:, i], ddq_pin[:, i]))
print('Shape of tau_pin:\t', np.array(tau_pin).shape)
tau_pin=np.array(tau_pin)
tau_pin=np.double(tau_pin)



#print('Shape of tau_test_q_than:\t', np.array(tau).shape)




# # ========== Step 4 - Create IDM with pinocchio
w = []  # Regression vector
w_pin=[]
for i in range(nbSamples):
    w_pin.extend(pin.computeJointTorqueRegressor(model, data, q_pin[:, i], dq_pin[:, i], ddq_pin[:, i]))
w_pin=np.array(w_pin)



for i in range(nbSamples):
     w.extend(pin.computeJointTorqueRegressor(model, data, q[:, i], dq[:, i], ddq[:, i]))
w=np.array(w)


## modifier W pour quelle contient dq et singe(dq) pour ajouter les parametres de frotements Fv et Fs
dq_stack=[]
dq_stack.extend(dq[0])
dq_stack.extend(dq[1])
dq_stack=np.array([dq_stack])
dq_stack=dq_stack.T

dq_stack_pin=[]
dq_stack_pin.extend(dq_pin[0])
dq_stack_pin.extend(dq_pin[1])
dq_stack_pin=np.array([dq_stack_pin])
dq_stack_pin=dq_stack_pin.T

# calcule de signe(dq)
dq_sign=np.sign(dq_stack)
dq_sign_pin=np.sign(dq_stack_pin)

w=np.concatenate([w,dq_stack], axis=1)
w=np.concatenate([w,dq_sign], axis=1)

w_pin=np.concatenate([w_pin,dq_stack_pin], axis=1)
w_pin=np.concatenate([w_pin,dq_sign_pin], axis=1)

#affichage
print('Shape of W_pin:\t',w_pin.shape)
print('Shape of dq_stack  :\t',dq_stack.shape)
print('Shape of W regresseur  :\t',w.shape)
print('\n')

## calcule pour une matrice definie positive
def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
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
        return True
    except np.linalg.LinAlgError:
        return False


p_pin=np.dot(w_pin.transpose(),w_pin)
q_pin= -np.dot(tau_pin.transpose(),w_pin)
P = np.dot(w.transpose(),w)
q = -np.dot(tau.transpose(),w)


P=nearestPD(P)
p_pin=nearestPD(p_pin)
print('je suis avant qp solver')

# #contrainte masse positive
# G=([-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#    [0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#    [0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#    [0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#    [0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0],
#    [0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0],
#    [0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0],
#    [0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0])
#h=[0,0,0,0,0,0,0,0]
# #contrainte masse (m1,m2) positive 0.15<mx,my,mz<0.3
# G=([-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#    [0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#    [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],   
#    [0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#    [0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#    [0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0],
#    [0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0],
#    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
#    [0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0],
#    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
#    [0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0],
#    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])

# h=[0,-0.15,0.3,-0.15,0.3,-0.15,0.3,0,-0.15,0.3,-0.15,0.3,-0.15,0.3]

#contrainte masse (m1,m2) positive 0.15<mx,my,mz<0.3 parametres de base >0 et frottements >0
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
   [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
   [0,0,0,-1,0,0,0,0,0,0,-0.25,0,0,0,0,0,0,0,0,0,0,0],
   [0,-1,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0],
   [0,0,0,0,0,0,-1,0,0,0,-0.3125,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0],
   )
G1=G
h=[0,-0.15,0.3,-0.15,0.3,-0.15,0.3,0,-0.15,0.3,-0.15,0.3,-0.15,0.3,0,0,0,0,0,0]
h1=[0,-0.2,0.3,-0.2,0.3,-0.2,0.3,0,-0.2,0.2,-0.2,0.3,-0.2,0.3,0,0,0,0,0,0]

G=np.array(G)
h=np.array(h)
G=np.double(G)
h=np.double(h)

G1=np.array(G1)
h1=np.array(h1)
G1=np.double(G1)
h1=np.double(h1)
#Any constraints that are >= must be multiplied by -1 to become a <=.
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
phi_etoile1=qpsolvers.solve_qp(
            P,
            q,
            G1,#G Linear inequality matrix.
            h1,#Linear inequality vector.
            A=None,
            b=None,
            lb=None,
            ub=None,
            solver="quadprog",
            initvals=None,
            sym_proj=True
            )


phi_etoile_pin=qpsolvers.solve_qp(
            p_pin,
            q_pin,
            G=None,
            h=None,
            
            A=None,
            b=None,
            lb=None,
            ub=None,
            solver="quadprog",
            initvals=None,
            sym_proj=True
            )

print('*****************************************')
print('phi_etoile',phi_etoile.shape)
print('phi_etoile_pin',phi_etoile_pin.shape)
print('*****************************************')

## calcule du torque estime 
tau_estime=np.dot(w,phi_etoile)
tau_estime1=np.dot(w,phi_etoile1)
tau_estime_pin=np.dot(w_pin,phi_etoile_pin)

# samples = []
# for i in range(20):
#         samples.append(i)
# # variation of the parameters
# # trace le resultat dans un graph 
# # les deux plot sur la memes figure
# plt.figure('phi et phi etoile')
# plt.plot(samples, phi, 'g', linewidth=1, label='phi')
# plt.plot(samples, phi_etoile, 'b:', linewidth=2, label='phi etoile')
# plt.plot(samples, phi_etoile_pin, 'r:', linewidth=1, label='phi etoile_sans Contraintes')
# plt.title('phi and phi etoile(M>0) and phi_etoile sans contraintes')
# plt.xlabel('20 Samples')
# plt.ylabel('parametres')
# plt.legend()
# plt.show()

## Plot the torques values
samples = []
for i in range(NQ*nbSamples):
        samples.append(i)

plt.figure('torque et torque estime')
plt.plot(samples, tau, 'g', linewidth=2, label='tau')
plt.plot(samples,tau_estime, 'b', linewidth=1.5, label='tau estime')
plt.plot(samples, tau_estime1, 'r', linewidth=1, label='tau estime 1')
plt.title('tau and tau_estime')
plt.xlabel('2000 Samples')
plt.ylabel('parametres')
plt.legend()
plt.show()

err = []
err1 = []
for i in range(nbSamples * NQ):
    err.append(abs(tau[i] - tau_estime[i]) * abs(tau[i] - tau_estime[i]))
    err1.append(abs(tau[i] - tau_estime1[i]) * abs(tau[i] - tau_estime1[i]))


# print(np.array(err).shape)
plt.plot(samples, err, linewidth=2, label="err")
plt.plot(samples, err1,linewidth=1, label="err1")
plt.title("erreur quadratique")
plt.legend()
plt.show()

# plt.figure('torque pin et torque pin estime')
# plt.plot(samples, tau_pin, 'g', linewidth=2, label='tau_pin')
# plt.plot(samples,tau_estime_pin, 'b', linewidth=1, label='tau_pin estime')
# # plt.plot(samples, tau_estime_pin, 'r:', linewidth=1, label='phi etoile_sans Contraintes')
# plt.title('tau pin tau_estime pin ')
# plt.xlabel('2000 Samples')
# plt.ylabel('parametres')
# plt.legend()
# plt.show()

'''
pour le calcule des paramètres standard il n y a pas besoin des paramètres de base (en tout cas poru le moment). Vous devez trouver Phi* (8x1) le vecteur contenant tous les paramètres inertiels.
Notez que si vous multipliez le regresseur R par phi vous obtenez tau=RPhi 

vous cherchez donc à déterminer Phi* qui minimise l’erreur quadratique ||tau_m-RPhi  ||^2 avec tau_m le couple mesuré (celui donné par Thanh). Déjà faites cela avec qp-solvers ensuite rajouter la contrainte que les masses (elements 1 et 5 du vecteur phi) M>=0
'''





# ========== Step 5 - Remove non dynamic effect columns then remove zero value columns then remove the parameters related to zero value columns at the end we will have a matix W_modified et Phi_modified
""""
threshold = 0.000001
W_modified = np.array(W[:])
tmp = []
for i in range(len(phi)):
    if (np.dot([W_modified[:, i]], np.transpose([W_modified[:, i]]))[0][0] <= threshold):
        tmp.append(i)
tmp.sort(reverse=True)

phi_modified = phi[:]
names_modified = names[:]
for i in tmp:
    W_modified = np.delete(W_modified, i, 1)
    phi_modified = np.delete(phi_modified, i, 0)
    names_modified = np.delete(names_modified, i, 0)

# print('shape of W_m:\t', W_modified.shape)
# print('shape of phi_m:\t', np.array(phi_modified).shape)

# ========== Step 6 - QR decomposition + pivoting

(Q, R, P) = sp.qr(W_modified, pivoting=True)

# P sort params as decreasing order of diagonal of R
# print('shape of Q:\t', np.array(Q).shape)
# print('shape of R:\t', np.array(R).shape)
# print('shape of P:\t', np.array(P).shape)

# ========== Step 7 - Calculate base parameters

tmp = 0
for i in range(len(R[0])):
    if R[i, i] > threshold:
        tmp = i

R1 = R[:tmp+1, :tmp+1]
R2 = R[:tmp+1, tmp+1:]

Q1 = Q[:, :tmp+1]

for i in (tmp+1, len(P)-1):
    names.pop(P[i])

# print('Shape of R1:\t', np.array(R1).shape)
# print('Shape of R2:\t', np.array(R2).shape)
# print('Shape of Q1:\t', np.array(Q1).shape)

beta = np.dot(np.linalg.inv(R1), R2)
# print('Shape of res:\t', beta.shape)

# beta = np.round(res, 6)
# print(res)

# ========== Step 8 - Calculate the Phi modified

phi_base = np.dot(np.linalg.inv(R1), np.dot(Q1.T, tau))  # Base parameters
W_base = np.dot(Q1, R1)                                  # Base regressor

# print('Shape of phi_m:\t', np.array(phi_modified).shape)
# print('Shape of W_m:\t', np.array(W_modified).shape)

inertialParameters = {names_modified[i]: phi_base[i]
                      for i in range(len(phi_base))}
print("Base parameters:\n", inertialParameters)


params_rsortedphi = [] # P donne les indice des parametre par ordre decroissant 
params_rsortedname=[]
for ind in P:
    params_rsortedphi.append(phi_modified[ind])
    params_rsortedname.append(names_modified[ind])

params_idp_val = params_rsortedphi[:tmp+1]
params_rgp_val = params_rsortedphi[tmp+1]
params_idp_name =params_rsortedname[:tmp+1]
params_rgp_name = params_rsortedname[tmp+1]
params_base = []
params_basename=[]

for i in range(tmp+1):
    if(beta[i] == 0):
        params_base.append(params_idp_val[i])
        params_basename.append(params_idp_name[i])

    else:
        params_base.append(str(params_idp_val[i]) + ' + '+str(round(float(beta[i]), 6)) + ' * ' + str(params_rgp_val))
        params_basename.append(str(params_idp_name[i]) + ' + '+str(round(float(beta[i]), 6)) + ' * ' + str(params_rgp_name))

print('base parameters and their identified values: \n')
print(params_base)
print('\n')
table = [phi_base,params_base]
print(table)
print('\n')
table1 = [params_basename,names_modified]
print('base_parametre and equation \n')
print(table1)
    # print('valeurs base et calcul\t',table[i][i])
# print('finale table shape \t', np.array(table).shape)
# print(table)



# ========== Step 9 - calcul de tau avec phi(paramaetre de base) et W_b le base regressor

tau_base = np.dot(W_base, phi_base)

samples = []
for i in range(nbSamples * NQ):
    samples.append(i)

# trace le resultat dans un graph
# les deux plot sur la memes figure
plt.plot(samples, tau_base, color='green', linewidth=2,
         label="tau_base")  # linewidth linestyle
plt.plot(samples, tau, color='blue', linewidth=1, label="tau")
plt.legend()
plt.title("graphe montrant tau et tau_base")

# les deux plot sur deux figures differentes
fig, axs = plt.subplots(2)
fig.suptitle('tau et tau_base separement')
axs[0].plot(samples, tau, color='blue', label="tau")
# plt.legend()
axs[1].plot(samples, tau_base, color='green', label="tau_base")
plt.legend()
# showing results
plt.show()

# ========== Step 10 - calcul phi_etoile moindre carre min abs(tau - phi_etoile * W_b)^2.On applique le raisonement de moindre carre classique en annulant le gradien de l'erreur avec une Hes>0
# print('W\t',np.array(W).shape,'\t W_base \t',W_base.shape)
# wtw=np.dot(W.T,W)
# # wtw=np.array(wtw)
# w_btw_b = np.linalg.inv(wtw)
# w_bt_tau = np.dot(W.T, tau)
# phi_etoile = np.dot(w_btw_b, w_bt_tau)
# print('shape of phi_etoile \t', phi_etoile.shape)
# affichage de phi et phi_etoile

# samples1 = []
# for i in range(phi_etoile.shape[0]):
#     samples1.append(i + 1)

# plt.scatter(samples1, phi_base, color='green', linewidth=2, label="phi(base)")
# plt.scatter(samples1, phi_etoile, color='red', linewidth=1, label="phi etoile")
# plt.title("graphe montrant phi et phi etoile")
# plt.legend()
# plt.show()

# ========== Step 11 - calcul l'err entre tau et tau_base calcule a partir de l'identification

err = []
for i in range(nbSamples * NQ):
    err.append(abs(tau[i] - tau_base[i]) * abs(tau[i] - tau_base)[i])

# print(np.array(err).shape)
plt.plot(samples, err, label="err")
plt.title("erreur quadratique")
plt.legend()
plt.show()

# print("press enter to continue")
# input()
gv.deleteNode('world', True)  # name, all=True

"""
