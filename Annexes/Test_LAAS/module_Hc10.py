import pinocchio as pin
from pinocchio import robot_wrapper
from pinocchio.visualize import GepettoVisualizer
from pinocchio import RobotWrapper
import os
# module pour importer le robot hc10 dans le gepettoviewer


def load_hc10_model(path = "",default=True):
# Fonction renvois le robotwrapper du hc10, l'affiche dans ça config initial
    if default==True:
        print('Attention à être dans le bon fichier !')
        path = os.getcwd()  + '/'
    else:
        print("Renseignez le chemin d'accé du package ou sont stockés les fichiers .stl et .dae")
    urdf = path + 'motoman_hc10_support/urdf/hc10_gazebo.urdf'
    hc10 = RobotWrapper.BuildFromURDF(urdf,path)
    hc10.initViewer(loadModel=True)
    hc10.display(hc10.q0)
    return hc10
