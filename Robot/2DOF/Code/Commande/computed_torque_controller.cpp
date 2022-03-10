#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include <stdio.h>
#include "utilis/test.hpp"
#include <Eigen/Dense>
using namespace std;
using namespace pinocchio;

Model model;
string urdf_file= "/home/jo/Documents/projet/project_M2/Robot/2DOF/Modeles/planar_2DOF/URDF/planar_2DOF_TCP.urdf" ;// change it for your own urdf path 

Eigen::MatrixXd b(2,2);
int main(void)
{
    pinocchio::urdf::buildModel(urdf_file,model);
    pinocchio::Data data(model);
    cout<<"model number of joint "<<model.njoints<<endl<<endl;
    cout<<"data omf joint 1"<<data.oMf[1]<<endl;
    cout<<"m "<<b<<endl; 
    
    //count();
    return 0;
}