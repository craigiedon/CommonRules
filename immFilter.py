from dataclasses import dataclass
from itertools import product
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class LinearModel:
    F: np.ndarray  # State Matrix
    G: np.ndarray  # Noise Matrix
    proc_noise_cov: np.ndarray  # Process Noise Covariance
    H: np.ndarray  # Measurement Projection Matrix
    measure_noise_cov: np.ndarray  # Measurement Noise Covariance


def c_vel_long_model(ts: float, q_p: float, q_m: float) -> LinearModel:
    return LinearModel(
        F=np.array([[1.0, ts, (ts ** 2.0) / 2.0],
                    [0.0, 1.0, ts],
                    [0.0, 0.0, 0.0]]),
        G=np.array([(ts ** 2.0) / 2.0, ts, 1.0]),
        proc_noise_cov=np.identity(3) * q_p,
        H=np.array([[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0]]),
        measure_noise_cov=np.identity(2) * q_m
    )


def c_acc_long_model(ts: float, q_p: float, q_m: float) -> LinearModel:
    return LinearModel(
        F=np.array([[1.0, ts, (ts ** 2.0) / 2.0],
                    [0.0, 1.0, ts],
                    [0.0, 0.0, 1.0]]),
        G=np.array([(ts ** 2.0) / 2.0, ts, 1.0]),
        proc_noise_cov=np.identity(3) * q_p,
        H=np.array([[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0]]),
        measure_noise_cov=np.identity(2) * q_m
    )


def run():
    x_0 = np.array([0.0, 30.0, 5.0])
    ts = 0.05
    q_p = 0.01
    q_m = 10.0

    # Longitudinal Models
    cv_lm = c_vel_long_model(ts, q_p, q_m)
    ca_lm = c_acc_long_model(ts, q_p, q_m)

    N = 50
    xs = []
    x_next = x_0
    for i in range(N - 1):
        x_next = update_longitude(x_next, ca_lm)
        xs.append(x_next)
    xs = np.array(xs)

    zs = np.array([(ca_lm.H @ x) + np.random.multivariate_normal(np.zeros(len(ca_lm.H)), ca_lm.measure_noise_cov) for x in xs])

    env_models = [(cv_lm, "Const Velocity", "green"),
                  (ca_lm, "Const Acceleration", "orange")]
    model_mus = []
    for m, m_name, _ in env_models:
        mus, covs = kalman_filter_batch(m, x_0, m.proc_noise_cov, zs)
        model_mus.append(mus)
        assert len(mus) == len(zs)

    # Initially uniform model prior
    model_priors = np.array([0.5, 0.5])

    # Rows = from mode, Cols = to mode
    # uniform model transitions
    m_trans_ps = np.array([[0.5, 0.5],
                                        [0.5, 0.5]])

    m_probs = model_priors
    # Evaluate mixing probabilities
    mix_prob = np.zeros((len(env_models), len(env_models)))
    for i, j in product(range(len(env_models)), repeat=2):
        print(i, j)
        mix_norm = sum([m_trans_ps[l][i] * m_probs[l] for l in range(len(env_models))])
        mix_prob[j][i] = m_trans_ps[i][j] * m_probs[j] / mix_norm

    print("Mix probs: ", mix_prob)

    mix_ests = []
    mix_covs = []

    for m in env_models:
        pass

    plt.scatter(np.arange(0, len(zs)), zs[:, 0], marker='x')
    plt.scatter(np.arange(0, len(xs)), xs[:, 0], color='black', facecolors='none', s=4.0)

    for (m, m_name, c), mus in zip(env_models, model_mus):
        plt.plot(np.arange(len(mus)), mus[:, 0], label=m_name, color=c)

    # plt.scatter(np.arange(0, N), xs[:, 0])
    plt.xlabel("Timestep")
    plt.ylabel("Position")
    plt.legend(loc='best')
    plt.show()


def kalman_filter_batch(env_model: LinearModel, mu_0, cov_0, zs) -> Tuple[np.ndarray, np.ndarray]:
    mus = []
    covs = []

    mu = mu_0
    cov = cov_0
    for z in zs:
        mu, cov = kalman_filter(env_model, mu, cov, z)
        mus.append(mu)
        covs.append(cov)

    return np.array(mus), np.array(covs)


def kalman_filter(env_model: LinearModel,
                  old_mu,
                  old_cov: np.ndarray,
                  z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred_mu, pred_cov = kalman_predict(env_model.F, old_mu, old_cov, env_model.proc_noise_cov)
    est_mu, est_cov = kalman_measure_update(pred_mu, pred_cov, env_model.H, env_model.measure_noise_cov, z)

    return est_mu, est_cov


def kalman_predict(A, old_mu, old_cov, proc_noise_v) -> Tuple[np.ndarray, np.ndarray]:
    pred_mu = A @ old_mu
    pred_cov = A @ old_cov @ A.T + proc_noise_v
    return pred_mu, pred_cov


def kalman_measure_update(pred_mu, pred_cov, C, measure_noise_v, z) -> Tuple[np.ndarray, np.ndarray]:
    kalman_gain = pred_cov @ C.T @ np.linalg.inv(C @ pred_cov @ C.T + measure_noise_v)
    est_mu = pred_mu + kalman_gain @ (z - C @ pred_mu)
    est_cov = (np.identity(len(kalman_gain)) - kalman_gain @ C) @ pred_cov

    return est_mu, est_cov


def update_longitude(x: np.ndarray, env_model: LinearModel) -> np.ndarray:
    noise = np.random.multivariate_normal(np.zeros(x.shape), env_model.proc_noise_cov)
    x_next = (env_model.F @ x) + (env_model.G @ noise)
    return x_next


def update_latitude(x: np.ndarray, F: np.ndarray, G: np.ndarray, noise_cov: np.ndarray,
                    affine_const: np.ndarray) -> np.ndarray:
    return None


if __name__ == "__main__":
    run()
