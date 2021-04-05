"""
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
    @Author: Aaron Guan (zhongg@andrew.cmu.edu), 2021
"""

import argparse
import time

import scipy.linalg
from scipy.sparse import csr_matrix

from solvers import *
from utils import *


def create_linear_system(odoms, observations, sigma_odom, sigma_observation, n_poses, n_landmarks):
    """
    :param odoms: (n_odom, 2). Odometry measurements between i and i+1 in the global coordinate system.
    :param observations: (n_obs, 4). Landmark measurements between pose i and landmark j in the global coordinate system.
            - observations[i, 0]: robot pose index in [0, n_poses), at which measurement was made.
            - observations[i, 1]: landmark index in [0, n_landmarks), which is being observed.
            - observations[i, 2:4]: (Δx,Δy) in the linear setup, (θ,r) in the nonlinear setup,
                                    from poses[observations[i, 0]] to landmark[observations[i, 1]]
    :param sigma_odom: (2, 2). Shared covariance matrix of odometry measurements.
    :param sigma_observation: (2, 2). Shared covariance matrix of landmark measurements.

    :return A (M, N) Jacobian matrix, b (M, ) Residual vector.
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

    M = (n_odom + 1) * 2 + n_obs * 2  # Added extra 1 for prior on the first pose
    N = n_poses * 2 + n_landmarks * 2

    A = np.zeros((M, N))
    b = np.zeros((M,))

    # Prepare Sigma^{-1/2}.
    sqrt_inv_odom = np.linalg.inv(scipy.linalg.sqrtm(sigma_odom))
    sqrt_inv_obs = np.linalg.inv(scipy.linalg.sqrtm(sigma_observation))

    # Jacobian of odmetry measurement function
    odomJ = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]])
    # Jacobian of landmark measurement function
    obsJ = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]])

    # First fill in the prior to anchor the 1st pose at (0, 0)
    A[0, 0] = 1
    A[1, 1] = 1

    # Then fill in odometry measurements
    for odom_idx in range(n_odom):
        # Update the (2, 4) pose sub-matrix for A, corresponding to the odom data id
        A[2 + odom_idx * 2: 2 + (odom_idx + 1) * 2, odom_idx * 2: odom_idx * 2 + 4] = sqrt_inv_odom @ odomJ
        # Update the (2, ) odom sub-rows for b, corresponding to the odom data id
        b[2 + odom_idx * 2: 2 + (odom_idx + 1) * 2] = (sqrt_inv_odom @ odoms[odom_idx].reshape(2, 1)).squeeze()

    # Then fill in landmark measurements
    obs_row_offset = (n_odom + 1) * 2
    for obs_idx in range(n_obs):
        pose_id, landmark_id = observations[obs_idx, :2].astype(int)
        measurement = observations[obs_idx, 2:]

        # Update the (2, 2) pose sub-matrix for A, corresponding to the pose id
        A[obs_row_offset + obs_idx * 2: obs_row_offset + (obs_idx + 1) * 2, pose_id * 2: pose_id * 2 + 2] \
            = sqrt_inv_obs @ obsJ[:, :2]
        # Update the (2, 2) landmark sub-matrix for A, corresponding to the landmark id
        A[obs_row_offset + obs_idx * 2: obs_row_offset + (obs_idx + 1) * 2,
        (n_poses + landmark_id) * 2: (n_poses + landmark_id + 1) * 2] = sqrt_inv_obs @ obsJ[:, 2:]
        # Update the (2, ) observation sub-rows for b, corresponding to the observation data id
        b[obs_row_offset + obs_idx * 2: obs_row_offset + (obs_idx + 1) * 2] \
            = (sqrt_inv_obs @ measurement.reshape(2, 1)).squeeze()

    return csr_matrix(A), b


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="../data/2d_linear.npz", help='path to npz file')
    parser.add_argument(
        '--method',
        nargs='+',
        choices=['default', 'pinv', 'qr', 'lu', 'qr_colamd', 'lu_colamd'],
        default=['pinv'],
        help='method')
    parser.add_argument(
        '--repeats',
        type=int,
        default=1,
        help=
        'Number of repeats in evaluation efficiency. Increase to ensure stablity.'
    )
    args = parser.parse_args()

    data = np.load(args.data)

    # Plot gt trajectory and landmarks for a sanity check.
    gt_traj = data['gt_traj']
    gt_landmarks = data['gt_landmarks']
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', label='gt trajectory')
    plt.scatter(gt_landmarks[:, 0],
                gt_landmarks[:, 1],
                c='b',
                marker='+',
                label='gt landmarks')
    plt.legend()
    plt.show()

    n_poses = len(gt_traj)
    n_landmarks = len(gt_landmarks)

    odoms = data['odom']
    observations = data['observations']
    sigma_odom = data['sigma_odom']
    sigma_landmark = data['sigma_landmark']

    # Build a linear system
    A, b = create_linear_system(odoms, observations, sigma_odom,
                                sigma_landmark, n_poses, n_landmarks)

    # Solve with the selected method
    for method in args.method:
        print(f'Applying {method}')

        total_time = 0
        total_iters = args.repeats
        for i in range(total_iters):
            start = time.time()
            x, R = solve(A, b, method)
            end = time.time()
            total_time += end - start
        print(f'{method} takes {total_time / total_iters}s on average')

        if R is not None:
            plt.spy(R)
            plt.show()

        traj, landmarks = devectorize_state(x, n_poses)

        # Visualize the final result
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)
