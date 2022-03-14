import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
from pinocchio.robot_wrapper import RobotWrapper


# chemin repertoire urdf
path = '/home/corentin/project_M2'
urdf = '/home/corentin/project_M2/robots/urdf/planar_2DOF.urdf'



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


values = []
for i in  range(3):
    #m = m + model.inertias[i].toDynamicParameters()
    #m.append(model.inertias[i].toDynamicParameters())
    print(model.inertias[i].toDynamicParameters()) #vecteur mx... link 
    for j in model.inertias[i].toDynamicParameters():
        values += [j]


names = []
for i in range (1,4):
    names += ['m'+str(i), 'mx'+str(i), 'my'+str(i), 'mz'+str(i), 'Ixx'+str(i), 'Ixy'+str(i), 'Iyy'+str(i), 'Izx'+str(i), 'Izy'+str(i), 'Izz'+str(i)]
    

print(values)
print(names)

d = {}
for i in range(30):
    d[names[i]] = values[i]

print(d)

tau = []

#generate output 100 samples
for i in range(100):
    tau += pin.rnea(model,data,q[i],dq[i],ddq[i])

nb_sample = 100
for i in range(nb_sample): 
    pin.computeJointTorqueRegressor(model,data,q[i],dq[i],ddq[i])


#print("\n\n HELLO WORLD")
# Step 1 : load a model 
# Input, Output, Model  => algorithm model fixed data (length ...) data = variable (joint angle)
# Step 2 generate intertial parameters (mass, first moment of inertial, Inertial matrix)
# Step 3 : generate input (joint angle, velocity, acceleration)
# Input/ZDOF : 100 samples -> (q, dq, ddq)
# Step 4 : Create IDM by pinocchio



# Step 3 : Grab a coffee
# Step 4 : Make some code
# Step 5 : Destroy the robot