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
for i in range(nb_sample): 
    pin.computeJointTorqueRegressor(model,data,q[i],dq[i],ddq[i])



# St: ep 3 : Grab a coffee
# Step 4 : Make some code
# Step 5 : Destroy the robot