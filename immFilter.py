from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def run():
    x_0 = np.array([0.0, 2.0, 0.0])
    ts = 1.0

    # Constant Velocity Longitudinal Model
    F_c = np.array([[1.0, ts, (ts ** 2.0) / 2.0],
                    [0.0, 1.0, ts],
                    [0.0, 0.0, 0.0]])

    G_c = np.array([(ts ** 2) / 2, ts, 1])

    # Constant Acceleration Longitudinal Model
    F_a = np.array([[1.0, ts, (ts ** 2.0) / 2.0],
                    [0.0, 1.0, ts],
                    [0.0, 0.0, 1.0]])

    G_a = np.array([(ts ** 2) / 2, ts, 1])

    # Process Noise Longitudinal
    q_p = 0.1
    proc_noise_cov = np.array([[q_p, 0.0, 0.0],
                               [0.0, q_p, 0.0],
                               [0.0, 0.0, q_p]])

    # Measurement Model
    H = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]])

    # Measurement Noise Longitudinal
    q_m = 10.0
    measure_noise_cov = np.identity(2) * q_m

    xs = [x_0]
    N = 50
    for i in range(N - 1):
        x_next = update_longitude(xs[-1], F_c, G_c, proc_noise_cov)
        xs.append(x_next)
        print("xn: ", x_next)
    xs = np.array(xs)

    zs = []
    for x in xs:
        zs.append(H @ x + np.random.multivariate_normal(np.zeros(len(H)), measure_noise_cov))
    zs = np.array(zs)

    # plt.show()

    mu_0 = x_0
    cov_0 = proc_noise_cov

    mus = [mu_0]
    covs = [cov_0]

    for z in zs:
        new_mu, new_cov = kalman_filter(F_c, H, proc_noise_cov, measure_noise_cov, mus[-1], covs[-1], z)
        mus.append(new_mu)
        covs.append(new_cov)

    mus = np.array(mus)
    covs = np.array(covs)

    plt.scatter(np.arange(0, len(zs)), zs[:, 0], marker='x')
    plt.scatter(np.arange(0, len(xs)), xs[:, 0], color='black', facecolors='none', s=4.0)
    plt.plot(np.arange(len(mus)), mus[:, 0], color='orange')

    # plt.scatter(np.arange(0, N), xs[:, 0])
    plt.xlabel("Timestep")
    plt.ylabel("Position")
    plt.show()


def kalman_filter(A: np.ndarray,
                  C: np.ndarray,
                  proc_noise_v: np.ndarray,
                  measure_noise_v: np.ndarray,
                  prev_mu,
                  prev_cov: np.ndarray,
                  z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred_mu = A @ prev_mu
    pred_cov = A @ prev_cov @ A.T + proc_noise_v

    inv_part = np.linalg.inv(C @ prev_cov @ C.T + measure_noise_v)
    meas_part = pred_cov @ C.T
    kalman_gain = meas_part @ inv_part
    est_mu = pred_mu + kalman_gain @ (z - C @ pred_mu)
    est_cov = (np.identity(len(kalman_gain)) - kalman_gain @ C) @ pred_cov

    return est_mu, est_cov


def update_longitude(x: np.ndarray, F: np.ndarray, G: np.ndarray, noise_cov: np.ndarray) -> np.ndarray:
    noise = np.random.multivariate_normal(np.zeros(x.shape), noise_cov)
    x_next = (F @ x) + (G @ noise)
    return x_next


def update_latitude(x: np.ndarray, F: np.ndarray, G: np.ndarray, noise_cov: np.ndarray,
                    affine_const: np.ndarray) -> np.ndarray:
    return None


if __name__ == "__main__":
    run()
