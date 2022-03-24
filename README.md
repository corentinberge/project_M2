# Command 

For the command we used pinocchio library to compute all the matrix needed for the command like the jacobian : 

    The jacobian is computed with respect of the frame, of the end effector and the reference base world
    all integration are made by using an approximation based on equation : u_k+1 = uk-1 + delta_T.dotuk
    

we have implemented a computed torque controller for the position problem ( we have encounter a problem to compute the orientational error ) this law works on 2D and on 3D only in position. 

you can find a proposition of a ROS implementation on another branch
