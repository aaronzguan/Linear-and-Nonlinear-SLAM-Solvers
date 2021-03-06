# Least-Square SLAM Problems - Linear and Nonlinear SLAM Solvers

This is the homework in CMU 16833-Robot Localization and Mapping Spring 2021. Different from the EKF SLAM, which is proactive, the offline SLAM algorithm can be viewed as a Least-Square problem: it simply accumulates information and then perform the inference to minimize the Mahalanobis Distance between the observation and estimation. 

For **linear system**, we can immediately get the results without iterations by solving the linear least-square problem |Ax-b|, where x is the final state vectors including both robots and landmarks. The only question is how to formulate the A and b to solve the least-square problem.

For **non-linear system**, First-order Taylor serier expansion is used to linearize the system and **Gauss-Newton** iterative algorithm is used to update the solution until convergence. Therefore, an state initialization of the system is required for the non-linear system, at which we linearize the system and update the state based on the initialization in each iteration. After linearization, we can still formulate the A, b for linear least-square problem |Ax-b| and use the same solver to get the state vector x. Note that here x is not final state vector but an update of our state from our previous state.

 We have all the odometry and observation data and iteratively optimize the non-linear SLAM problem to get the states of robot pose and landmarks all in once:

![least_sqaure](http://www.sciweavers.org/upload/Tex2Img_1617660937/render.png)

We convert from the Mahalanobis Distance to the Euclidean Square Distance (L2) such that we can formulate the least-square problem to get our state vector. The A matrix and b would be:

![A](http://www.sciweavers.org/upload/Tex2Img_1617661049/render.png), epsilon is the covariance, H is the Jacobian matrix.

![b](http://www.sciweavers.org/upload/Tex2Img_1617661097/render.png), z_i is the observation, h is the estimation


In this homework, three solvers are used to solve the Normal Equation

A^TAx = A^Tb


1. Pseudo-Inverse:
   
   x = (A^T A)^{-1} A^T b
   

2. LU decomposition:
   
   x = (A^T A)^{-1} A^T b  = (LU)^{-1} A^T b
   

3. QR decomposition:

   |Ax - b|^2 = |Rx - z|^2 + |e|^2

Because the state vector includes both robot poses and landmarks position, therefore the MxN Jacobian matrix would be:

```
                |<---  Pose --->|<-- Landmark -->|
          H =  [[dh1/dx1, dh1/dx2, ...., dh1/dxn],
                [dh2/dx1, dh2/dx2, ...., dh2/dxn],
                ....
                [dhm/dx1, dhm/dx2, ..., dhm/dxn]]
```


## Academic Integrity Policy

Students at Carnegie Mellon are expected to produce their own original academic work. Please do not copy the code for your homework and do not violate academic integrity policy!
