"""
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
    @Author: Aaron Guan (zhongg@andrew.cmu.edu), 2021

    This is an 2D non-linear SLAM optimization implementation. We have all the
    odometry and observation data and iteratively optimize the non-linear SLAM
    ||Ax - b||^2 problem to get the states of robot pose and landmarks all in once..

    It utilizes the sparse matrix of A to efficiently get the solution.
    Different solves is used:
    1. Pseudo-inverse (x = (A^T A)^-1 A^T b)
    2. LU decomposition (x = (A^T A)^-1 A^T b  = (LU)^-1 A^T b)
    3. QR decomposition (|Ax - b|^2 = |Rx - z|^2 + |e|^2)
"""

import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import argparse
import matplotlib.pyplot as plt
from solvers import *
from utils import *


def warp2pi(angle_rad):
    """
    Warps an angle in [-pi, pi]. Used in the update step.
    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    angle_rad = angle_rad - 2 * np.pi * np.floor(
        (angle_rad + np.pi) / (2 * np.pi))
    return angle_rad


def init_states(odoms, observations, n_poses, n_landmarks):
    """
    Initialize the state vector given odometry and observations.
    """
    traj = np.zeros((n_poses, 2))
    landmarks = np.zeros((n_landmarks, 2))
    landmarks_mask = np.zeros((n_landmarks), dtype=np.bool)

    for i in range(len(odoms)):
        traj[i + 1, :] = traj[i, :] + odoms[i, :]

    for i in range(len(observations)):
        pose_idx = int(observations[i, 0])
        landmark_idx = int(observations[i, 1])

        if not landmarks_mask[landmark_idx]:
            landmarks_mask[landmark_idx] = True

            pose = traj[pose_idx, :]
            theta, d = observations[i, 2:]

            landmarks[landmark_idx, 0] = pose[0] + d * np.cos(theta)
            landmarks[landmark_idx, 1] = pose[1] + d * np.sin(theta)

    return traj, landmarks


def odometry_estimation(x, i):
    """
    Return odometry estimation given State vector at which we linearize the system.
    h = [dx, dy] = [rx_t+1 - rx_t, ry_t+1 - ry_t]

    :param x: State vector containing both the pose and landmarks
    :param i: Index of the pose to start from (odometry between pose i and i+1)
    :return odom: Odometry (\delta_x, \delta_y) in the shape (2, )
    """
    odom = x[i * 2 + 2: i * 2 + 4] - x[i * 2: i * 2 + 2 ]
    return odom


def bearing_range_estimation(x, i, j, n_poses):
    """
    Return bearing range estimations given State vector at which we linearize the system.
    h = [theta, d] = [arctan2(ly - ry, lx - rx), sqrt((ly - ry)^2 + (lx - rx)^2)]

    :param x: State vector containing both the pose and landmarks
    :param i: Index of the pose to start from
    :param j: Index of the landmark to be measured
    :param n_poses: Number of poses
    :return obs: Observation from pose i to landmark j (theta, d) in the shape (2, )
    """
    obs = np.zeros((2,))
    rx, ry = x[i * 2: i * 2 + 2]
    lx, ly = x[(n_poses + j) * 2: (n_poses + j) * 2 + 2]
    obs[0] = np.arctan2(ly - ry, lx - rx)
    obs[1] = np.sqrt((lx - rx)**2 + (ly - ry)**2)
    return obs


def compute_meas_obs_jacobian(x, i, j, n_poses):
    """
    Compute Jacobian matrix of landmark measurement function at the provided linearization point
    J = [[dtheta/drx, dtheta/dry, dtheta/dlx, dtheta/dly],
        [dD/drx, dD/dry, dD/dlx, dD/dly]]

    :param x: State vector containing both the pose and landmarks
    :param i: Index of the pose to start from
    :param j: Index of the landmark to be measured
    :param n:_poses Number of poses
    :return jacobian: Derived Jacobian matrix in the shape (2, 4)
    """
    rx, ry = x[i * 2: i * 2 + 2]
    lx, ly = x[(n_poses + j) * 2: (n_poses + j) * 2 + 2]
    delta_x, delta_y = lx - rx, ly - ry
    q = delta_x**2 + delta_y**2
    jacobian = np.array([[delta_y/q, -delta_x/q, -delta_y/q, delta_x/q],
                         [-delta_x / np.sqrt(q), -delta_y / np.sqrt(q), delta_x / np.sqrt(q), delta_y / np.sqrt(q)]])
    return jacobian


def create_linear_system(x, odoms, observations, sigma_odom, sigma_observation,
                         n_poses, n_landmarks):
    """
    Formulate the linear system to optimize ||Ax - b||^2
    and return A, b then later uses different solvers to get solution.

    :param x: State vector x at which we linearize the system.
    :param odoms: Odometry measurements between i and i+1 in the global coordinate system. Shape: (n_odom, 2).
    :param observations: Landmark measurements between pose i and landmark j in the global coordinate system. Shape: (n_obs, 4).
    :param sigma_odom: Shared covariance matrix of odometry measurements. Shape: (2, 2).
    :param sigma_observation: Shared covariance matrix of landmark measurements. Shape: (2, 2).

    :return A: (M, N) Jacobian matrix.
    :return b: (M, ) Residual vector.
    where M = (n_odom + 1) * 2 + n_obs * 2, total rows of measurements.
          N = n_poses * 2 + n_landmarks * 2, length of the state vector.
          Jacobian Matrix
                |<---  Pose --->|<-- Landmark -->|
          H =  [[dh1/dx1, dh1/dx2, ...., dh1/dxn],
                [dh2/dx1, dh2/dx2, ...., dh2/dxn],
                ....
                [dhm/dx1, dhm/dx2, ..., dhm/dxn]]
    """

    n_odom = len(odoms)
    n_obs = len(observations)

    M = (n_odom + 1) * 2 + n_obs * 2
    N = n_poses * 2 + n_landmarks * 2

    A = np.zeros((M, N))
    b = np.zeros((M,))

    sqrt_inv_odom = np.linalg.inv(scipy.linalg.sqrtm(sigma_odom))
    sqrt_inv_obs = np.linalg.inv(scipy.linalg.sqrtm(sigma_observation))

    # Jacobian of odmetry measurement function.
    # Because odometry measurement function is linear, its Jacobian is also constant
    odomJ = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]])

    # First fill in the prior to anchor the 1st pose at (0, 0)
    A[0, 0] = 1
    A[1, 1] = 1

    # Then fill in odometry measurements
    for odom_idx in range(n_odom):
        # Get the real odometry measurement data
        odom_meas = odoms[odom_idx]
        # Get the estimated odometry data from State vector at which we linearize the system.
        odom_est = odometry_estimation(x, odom_idx)

        # Update the (2, 4) pose sub-matrix for A, A = sqrt(sigma)H
        A[2 + odom_idx * 2: 2 + (odom_idx + 1) * 2, odom_idx * 2: odom_idx * 2 + 4] = sqrt_inv_odom @ odomJ

        # Update the (2, ) rows for b, b = sqrt(sigma)(z - h)
        odom_diff = odom_meas - odom_est
        b[2 + odom_idx * 2: 2 + (odom_idx + 1) * 2] = (sqrt_inv_odom @ odom_diff.reshape(2, 1)).squeeze()

    # Then fill in landmark measurements
    obs_row_offset = (n_odom + 1) * 2
    for obs_idx in range(n_obs):
        pose_id, landmark_id = observations[obs_idx, :2].astype(int)
        # Get the real landmark measurement data
        obs_meas = observations[obs_idx, 2:]
        # Get the estimated observation data from State vector at which we linearize the system
        obs_est = bearing_range_estimation(x, pose_id, landmark_id, n_poses)

        # Compute Jacobian matrix of landmark measurement function at the provided linearization point
        obsJ = compute_meas_obs_jacobian(x, pose_id, landmark_id, n_poses)

        # Update the (2, 2) pose sub-matrix for A
        A[obs_row_offset + obs_idx * 2: obs_row_offset + (obs_idx + 1) * 2, pose_id * 2: pose_id * 2 + 2] \
            = sqrt_inv_obs @ obsJ[:, :2]
        # Update the (2, 2) landmark sub-matrix for A
        A[obs_row_offset + obs_idx * 2: obs_row_offset + (obs_idx + 1) * 2,
        (n_poses + landmark_id) * 2: (n_poses + landmark_id + 1) * 2] = sqrt_inv_obs @ obsJ[:, 2:]

        # Update the (2, ) observation sub-rows for b, corresponding to the observation data id
        obs_diff = obs_meas - obs_est
        obs_diff[0] = warp2pi(obs_diff[0])
        b[obs_row_offset + obs_idx * 2: obs_row_offset + (obs_idx + 1) * 2] \
            = (sqrt_inv_obs @ obs_diff.reshape(2, 1)).squeeze()

    return csr_matrix(A), b


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../data/2d_nonlinear.npz')
    parser.add_argument(
        '--method',
        nargs='+',
        choices=['default', 'pinv', 'qr', 'lu', 'qr_colamd', 'lu_colamd'],
        default=['default'],
        help='method')
    parser.add_argument(
        '--iter',
        type=int,
        default=10,
        help=
        'Number of iterations for solving non-linear optimization.'
    )

    args = parser.parse_args()

    data = np.load(args.data)

    # Plot gt trajectory and landmarks for a sanity check.
    gt_traj = data['gt_traj']
    gt_landmarks = data['gt_landmarks']
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-')
    plt.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], c='b', marker='+')
    plt.show()

    n_poses = len(gt_traj)
    n_landmarks = len(gt_landmarks)

    odom = data['odom']
    observations = data['observations']
    sigma_odom = data['sigma_odom']
    sigma_landmark = data['sigma_landmark']

    # Initialize: non-linear optimization requires a good init.
    for method in args.method:
        print(f'Applying {method}')
        traj, landmarks = init_states(odom, observations, n_poses, n_landmarks)
        print('Before optimization')
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)

        # Iterative optimization
        x = vectorize_state(traj, landmarks)
        total_iters = args.iter
        for i in range(total_iters):
            A, b = create_linear_system(x, odom, observations, sigma_odom,
                                        sigma_landmark, n_poses, n_landmarks)
            dx, _ = solve(A, b, method)
            x = x + dx.squeeze()
        traj, landmarks = devectorize_state(x, n_poses)
        print('After optimization')
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)
