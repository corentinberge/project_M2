import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
from pinocchio.robot_wrapper import RobotWrapper


# chemin repertoir urdf
path = '/home/jo/'
urdf = '/home/jo/robots/planar_2DOF/urdf/planar_2DOF.urdf'


print(path)
print(urdf)


robot = RobotWrapper()

#robot.BuildFromURDF(urdf,path,verbose=True)
robot.initFromURDF(urdf,path,verbose=True)
#print("MODEL DU ROBOT\n",robot.model)
#print("VISUAL MODEL ",robot.visual_model)

robot.initDisplay(loadModel=True)

data = robot.data
model = robot.model
#robot.inert
#print(data)
print(model)



for i in  range(3):
    #m = m + model.inertias[i].toDynamicParameters()
    #m.append(model.inertias[i].toDynamicParameters())
    print(model.inertias[i].toDynamicParameters()) #vecteur mx... link 

values = []
names = []
for i in range (3):
    names += ['m'+str(i), 'mx'+str(i), 'my'+str(i), 'mz'+str(i), 'Ixx'+str(i), 'Ixy'+str(i), 'Iyy'+str(i), 'Izx'+str(i), 'Izy'+str(i), 'Izz'+str(i)]
    values += model.inertias[i].toDynamicParameters()

d = {}
for i in range(3):
    for name in names,value in values[i]:
            d.put(name, value)



#print("\n\n HELLO WORLD")
# Step 1 : load a model 
# Input, Output, Model  => algorithm model fixed data (length ...) data = variable (joint angle)
# Step 2 generate intertial parameters (mass, first moment of inertial, Inertial matrix)
# Step 3 : generate input (joint angle, velocity, acceleration)
# Input/ZDOF : 100 samples -> (q, dq, ddq)
# Step 4 : Create IDM by pinocchio
nb_sample = 100
W = np.array(200,20)
for i in range(nb_sample): 
    w[:,i:i+2] = pin.computeJointTorqueRegressor(model,data,q[i],dq[i],ddq[i]) #surement faux 

#we need to remove zeros collumn np.dot(A^t,A) <= eps = 10^-6  => W' = regressors matrix with only parameter wich have an effect 
# QR pivoting (with scipy) sc.linalg.qr(W,pivoting=True) = Q,R,P on calcul les valeurs avec un seuil = 10-6 
# idenp_par = col> eps ; dep_par = col < eps  => Q = [Q1 Q2] R = [R1 R2]
# base param Beta = R1^-1R2  
# Wb = Q1*R1 independ_par[i]

# Generation output joint + 100 sample 

for i in range(nb_sample)
    torque = pin.rnea(model,data,q[i],dq[i],ddq[i])

# St: ep 3 : Grab a coffee
# Step 4 : Make some code
# Step 5 : Destroy the robot