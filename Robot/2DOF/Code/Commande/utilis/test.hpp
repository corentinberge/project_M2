#include <stdio.h>
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include <Eigen/QR>    

/*** This File is taking care of all useful function ***/
void robotDynamic(pinocchio::Model model,auto input,auto q,auto vq,auto aq,float dt):
{
    /** 
    Dynamic of the robot calculator for postion/speed control 
    tau =  input + G

    tau = J't.f
    ------------------------------
    IN
    
    robot   : a RobotWrapper object needed to compute gravity torque and other parameters
    input   : input signal of the function equals to B*deltaDotQ-K*deltaQ
    q       : current joints angles values
    vq      : current joints velocities 
    aq      : current joints acceleration values 
    dt      : time step between each execution of this function
    ---------------------------------
    OUT

    q : calculated joint angles values 
    dq : calculated joint velocities values 
    aq : calculated joint acceleration values 
    f : the force exerted by the manipulator 



    system : 
            Xp = Ax + Bu
            Y = x
            with u = tau, x = [q,vq], Xp = [vq,aq]
    **/

    A = pinocchio::crba(model,data,q); // compute mass matrix
    H = pinocchio::rnea(model,data,q,vq,np.zeros(aq.shape));  // compute dynamic drift -- Coriolis, centrifugal, gravity
    
    auto X = np.array([q,vq]);
    auto Xp = np.array([vq,A.completeOrthogonalDecomposition().pseudoInverse()*(input-H)]);
    X += Xp*dt;
}

void count()
{
    for(int i=0;i<11;i++)
        std::cout<<i<<std::endl;
}