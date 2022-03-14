from numpy import linalg
from numpy.core.fromnumeric import shape
from numpy.lib.nanfunctions import _nanmedian_small
from pinocchio.visualize import GepettoVisualizer
from pinocchio.robot_wrapper import RobotWrapper
import matplotlib.pyplot as plt
import scipy.linalg as sp
import pinocchio as pin
import numpy as np
import os

# urdf directory path
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

# print('shape of phi:\t', np.array(phi).shape)

# ========== Step 3 - Generate input and output - 100 samples

nbSamples = 1000  # number of samples

# Generate 100 inputs
q = np.random.rand(NQ, nbSamples) * np.pi - np.pi/2  # -pi/2 < q < pi/2
dq = np.random.rand(NQ, nbSamples) * 10              # 0 < dq  < 10
ddq = np.ones((NQ, nbSamples))                       # ddq = 1

tau = []
for i in range(nbSamples):
    tau.extend(pin.rnea(model, data, q[:, i], dq[:, i], ddq[:, i]))
# print('Shape of tau:\t', np.array(tau).shape)

# ========== Step 4 - Create IDM with pinocchio

W = []  # Regression vector
for i in range(nbSamples):
    W.extend(pin.computeJointTorqueRegressor(
        model, data, q[:, i], dq[:, i], ddq[:, i]))
# print('Shape of W:\t', np.array(W).shape)

# ========== Step 5 - Remove non dynamic effect columns then remove zero value columns then remove the parameters related to zero value columns at the end we will have a matix W_modified et Phi_modified

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

print('shape of W_m:\t', W_modified.shape)
print('shape of phi_m:\t', np.array(phi_modified).shape)

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
    if beta[i] == 0:
        params_base.append(params_idp_val[i])
        params_basename.append(params_idp_name[i])

    else:
        params_base.append(str(params_idp_val[i]) + ' + '+str(round(float(beta[i]), 6)) + ' * ' + str(params_rgp_val))
        params_basename.append(str(params_idp_name[i]) + ' + '+str(round(float(beta[i]), 6)) + ' * ' + str(params_rgp_name))
print('\n')
print('base parameters and their identified values:')
print(params_basename)
print(params_base)
print('\n')




# ========== Step 9 - calcul de tau avec phi(paramaetre de base) et W_b le base regressor
print(phi_base)
print('w de base ',np.array(W_base).shape)
print('phi de base ',np.array(phi_base).shape)
tau_base = np.dot(W_base, phi_base)
print('tau de base',np.array(tau_base).shape)
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

w_btw_b = np.linalg.inv(np.dot(W_base.T, W_base))
w_bt_tau = np.dot(W_base.T, tau)
phi_etoile = np.dot(w_btw_b, w_bt_tau)
print('shape of phi_etoile \t', phi_etoile.shape)

# affichage de phi et phi_etoile
samples1 = []
for i in range(phi_etoile.shape[0]):
    samples1.append(i + 1)

plt.scatter(samples1, phi_base, color='green', linewidth=2, label="phi(base)")
plt.scatter(samples1, phi_etoile, color='red', linewidth=1, label="phi etoile")
plt.title("graphe montrant phi et phi etoile")
plt.legend()
plt.show()

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
