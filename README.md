# Least-Square SLAM Problems - Linear and Nonlinear SLAM Solvers

This is the homework in CMU 16833-Robot Localization and Mapping Spring 2021. Different from the EKF SLAM, which is proactive, the offline SLAM algorithm can be viewed as a Least-Square problem: it simply accumulates information and then perform the inference to minimize the Mahalanobis Distance between the observation and estimation. 

 We have all the odometry and observation data and iteratively optimize the non-linear SLAM problem to get the states of robot pose and landmarks all in once:
$$
\theta = argmin \sum_i^N||h_i(\theta) - z_i||_\epsilon^2 = argmin||Ax - b||
$$
We convert from the Mahalanobis Distance to the Euclidean Square Distance (L2) such that we can formulate the least-square problem to get our state vector. The A matrix and b would be:
$$
A = \epsilon^{-1/2}H, \epsilon \text{ is the covariance, H is the Jacobian matrix}\\
b = \epsilon^{-1/2}(z_i - h_i(x)), z_i \text{  is the observation, h is the estimation}
$$
In this homework, three solvers are used to solve the Normal Equation
$$
A^TAx = A^Tb
$$

1. Pseudo-Inverse:
   $$
   x = (A^T A)^{-1} A^T b
   $$
   

2. LU decomposition:
   $$
   x = (A^T A)^{-1} A^T b  = (LU)^{-1} A^T b
   $$
   

3. QR decomposition:
   $$
   |Ax - b|^2 = |Rx - z|^2 + |e|^2
   $$

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