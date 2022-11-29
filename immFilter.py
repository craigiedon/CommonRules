from dataclasses import dataclass
from itertools import product
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from scipy.special import logsumexp
from scipy.stats import norm, multivariate_normal


@dataclass
class AffineModel:
    ts: float
    F: np.ndarray  # State Matrix
    G: np.ndarray  # Noise Matrix
    E: np.ndarray  # Affine Shift
    proc_noise_cov: np.ndarray  # Process Noise Covariance
    H: np.ndarray  # Measurement Projection Matrix
    measure_noise_cov: np.ndarray  # Measurement Noise Covariance


@dataclass
class StateFeedbackModel(AffineModel):
    k_p: float
    k_d: float


@dataclass
class KalmanResult:
    pred_mu: np.ndarray
    pred_cov: np.ndarray
    est_mu: np.ndarray
    est_cov: np.ndarray
    residual: np.ndarray
    S: np.ndarray


def c_vel_long_model(ts: float, q_p: float, q_m: float) -> AffineModel:
    return AffineModel(
        ts=ts,
        F=np.array([[1.0, ts, (ts ** 2.0) / 2.0],
                    [0.0, 1.0, ts],
                    [0.0, 0.0, 0.0]]),
        G=np.array([(ts ** 2.0) / 2.0,
                    ts,
                    1.0]),
        E=np.zeros(3),
        proc_noise_cov=np.identity(3) * q_p,
        H=np.array([[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0]]),
        measure_noise_cov=np.identity(2) * q_m
    )


def c_acc_long_model(ts: float, q_p: float, q_m: float) -> AffineModel:
    return AffineModel(
        ts=ts,
        F=np.array([[1.0, ts, (ts ** 2.0) / 2.0],
                    [0.0, 1.0, ts],
                    [0.0, 0.0, 1.0]]),
        G=np.array([(ts ** 2.0) / 2.0,
                    ts,
                    1.0]),
        E=np.zeros(3),
        proc_noise_cov=np.identity(3) * q_p,
        H=np.array([[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0]]),
        measure_noise_cov=np.identity(2) * q_m
    )


def lat_model(ts: float, k_d: float, k_p: float, p_ref: float, q_p: float, q_m: float) -> AffineModel:
    return StateFeedbackModel(
        ts=ts,
        F=np.array([
            [1.0, ts],
            [-ts * k_p, 1 - ts * k_d],
        ]),
        G=np.array([(ts ** 2.0) / 2.0, ts]),
        E=np.array([0.0, ts * k_p * p_ref]),
        proc_noise_cov=np.identity(2) * q_p,
        H=np.array([[1.0, 0.0]]),
        measure_noise_cov=np.identity(1) * q_m,
        k_p=k_p,
        k_d=k_d
    )


def kf_to_filterpy(model: AffineModel, x_0: np.ndarray) -> KalmanFilter:
    f = KalmanFilter(dim_x=len(x_0), dim_z=len(model.H @ x_0), dim_u=len(model.E))
    f.x = x_0
    f.F = model.F

    # Hack for Affine shift --- Put in the control?
    f.B = model.E

    f.H = model.H
    f.Q = np.diag(model.G) @ model.proc_noise_cov @ np.diag(model.G.T)
    f.R = model.measure_noise_cov

    return f


def run():
    N = 200
    ts = 0.05

    # Longitudinal Models
    x_long_0 = np.array([0.0, 10.0, 5.0])
    q_p = 1.0
    q_m = 1.0
    long_models = [c_vel_long_model(ts, q_p, q_m), c_acc_long_model(ts, q_p, q_m)]

    # Lat Models
    lane_positions = [-3.0, 0.0, 3.0]
    # lat_models: List[StateFeedbackModel] = [lat_model(ts, kd, 4.0, p_ref, q_p, q_m) for kd in np.linspace(3.0, 5.0, 3)
    #                                         for p_ref in lane_positions]
    lat_models = [lat_model(ts, 4.0, 4.0, lane_positions[0], q_p, q_m),
                  lat_model(ts, 4.0, 4.0, lane_positions[1], q_p, q_m)]

    # for p_ref in lane_positions:
    #     for kd in np.linspace(3.0, 5.0, 3):
    #         lat_models.append(lat_model(ts, kd, 4.0, p_ref, 0.0, 0.0))

    # Lat Simulation
    # Start in right lane, then at 3 seconds, lange change into middle
    x_lat_0 = np.array([lane_positions[0], 0.0])
    # lane_durations = [(3.0, 0), (7.0, 1)]
    lane_durations = [(10.0, 0)]
    xs_lat = []
    x_lat_next = x_lat_0
    for duration, lane_idx in lane_durations:
        xs_lat.extend(
            full_sim(x_lat_next, lat_model(ts, 4.0, 4.0, lane_positions[lane_idx], 0.0, 0.0), int(duration / ts)))
        x_lat_next = xs_lat[-1]
    xs_lat = np.array(xs_lat)
    assert len(xs_lat) == N

    ref_lat_model = lat_model(ts, 4.0, 4.0, lane_positions[0], 1.0, 0.0)
    zs_lat = np.array(
        [(ref_lat_model.H @ x) + np.random.multivariate_normal(np.zeros(len(ref_lat_model.H)),
                                                               ref_lat_model.measure_noise_cov) for x in
         xs_lat])

    lat_mus, lat_covs, mps_lat = imm_batch(lat_models,
                                           sticky_m_trans(len(lat_models), 0.95),
                                           unif_cat_prior(len(lat_models)),
                                           np.tile(x_lat_0, (len(lat_models), 1)),
                                           np.tile(np.identity(len(x_lat_0)), (len(lat_models), 1, 1)), zs_lat)

    assert len(lat_mus) == len(zs_lat)

    # Sanity Check with FilterPy
    f_lat_1 = kf_to_filterpy(lat_models[1], x_lat_0)

    sanity_mus_lat = []
    for z in zs_lat:
        f_lat_1.predict(np.array([1.0,1.0]), B=f_lat_1.B)
        f_lat_1.update(z)
        sanity_mus_lat.append(f_lat_1.x)

    sanity_mus_lat = np.array(sanity_mus_lat)

    assert len(lat_mus) == len(sanity_mus_lat)

    fig, axs = plt.subplots(1, 2)
    axs[0].scatter(ts * np.arange(len(xs_lat)), xs_lat[:, 0], color='black', facecolors='none', s=4.0)
    axs[0].scatter(ts * np.arange(len(zs_lat)), zs_lat, marker='x')
    axs[0].plot(ts * np.arange(len(lat_mus)), lat_mus[:, 0], color='orange')
    axs[0].plot(ts * np.arange(len(sanity_mus_lat)), sanity_mus_lat[:, 0], color='pink')

    axs[0].plot(ts * np.arange(len(xs_lat)), np.repeat(0.0, len(xs_lat)), linestyle='--', color='black', alpha=0.2)
    axs[0].plot(ts * np.arange(len(xs_lat)), np.repeat(3.0, len(xs_lat)), linestyle='--', color='black', alpha=0.2)
    axs[0].plot(ts * np.arange(len(xs_lat)), np.repeat(-3.0, len(xs_lat)), linestyle='--', color='black', alpha=0.2)
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Latitude (m)")
    axs[0].set_ylim(-5, 5)

    axs[1].plot(ts * np.arange(len(mps_lat)), mps_lat[:, 0], label='0')
    axs[1].plot(ts * np.arange(len(mps_lat)), mps_lat[:, 1], label='1')
    axs[1].legend()
    plt.show()

    # Long Simulation
    xs_long = []
    x_long_next = x_long_0
    simulation_model = c_acc_long_model(ts, 0.0, 1.0)
    for i in range(0, N // 2):
        x_long_next = update_sim(x_long_next, simulation_model)
        xs_long.append(x_long_next)

    # Car changes acceleration
    x_long_next[2] = -40
    for i in range(N // 2, N - 1):
        x_long_next = update_sim(x_long_next, simulation_model)
        xs_long.append(x_long_next)

    xs_long = np.array(xs_long)

    zs_long = np.array(
        [(simulation_model.H @ x) + np.random.multivariate_normal(np.zeros(len(simulation_model.H)),
                                                                  simulation_model.measure_noise_cov) for x in
         xs_long])


    # Single Model Estimations
    # single_model_pred_mus = []
    # single_model_est_mus = []
    # single_model_pred_covs = []
    # for m in long_models:
    #     pred_mus, pred_covs, est_mus, est_covs = kalman_filter_batch(m, x_long_0, m.proc_noise_cov, zs_long)
    #     single_model_pred_mus.append(pred_mus)
    #     single_model_est_mus.append(est_mus)
    #     single_model_pred_covs.append(pred_covs)
    #     print()
    #     assert len(est_mus) == len(zs_long)

    # print(f"Sanity First Pred Mu: {sanity_pred_mus[0]}")
    # print(f"My First Pred Mu: {single_model_pred_mus[0][0]}")
    #
    # print(f"Sanity First Pred Cov: {sanity_pred_covs[0]}")
    # print(f"My first Pred Cov: {single_model_pred_covs[0][0]}")

    # IMM Estimation
    fused_mus, fused_covs, model_prob_time = imm_batch(long_models, np.array([[0.5, 0.5], [0.5, 0.5]]),
                                                       np.array([0.5, 0.5]), np.tile(x_long_0, (len(long_models), 1)),
                                                       np.tile(np.identity(3), (len(long_models), 1, 1)), zs_long)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(np.arange(0, len(zs_long)), zs_long[:, 0], marker='x')
    ax.scatter(np.arange(0, len(xs_long)), xs_long[:, 0], color='black', facecolors='none', s=4.0)

    # ax.plot(np.arange(len(zs_long)), single_model_est_mus[0][:, 0], color='green', label='const v')
    # ax.plot(np.arange(len(zs_long)), single_model_est_mus[1][:, 0], color='pink', label='const a')

    ax.plot(np.arange(len(zs_long)), fused_mus[:, 0], color='orange', label='imm')

    # axs[0].plot(np.arange(len(zs)), sanity_mus[:, 0], color='brown', label='sanity mus')

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Position")
    ax.legend(loc='best')

    # axs[1].plot(np.arange(len(model_prob_time)), model_prob_time[:, 0], label='const v')
    # axs[1].plot(np.arange(len(model_prob_time)), model_prob_time[:, 1], label='const a')
    # axs[1].set_ylim(0.0 - 1e-2, 1.0 + 1e-2)
    # axs[1].legend(loc='best')
    # axs[1].set_xlabel("Timestep")
    # axs[1].set_ylabel("Model Probability")

    # for (m, m_name, c), mus in zip(env_models, model_mus):
    #     plt.plot(np.arange(len(mus)), mus[:, 0], label=m_name, color=c)

    plt.show()


def unif_m_trans(m: int) -> np.ndarray:
    return np.full((m, m), 1.0 / m)


def sticky_m_trans(m: int, stick_prob: float) -> np.ndarray:
    assert 0.0 < stick_prob <= 1.0
    if m == 1:
        return unif_m_trans(m)
    trans_prob = (1.0 - stick_prob) / (m - 1)
    trans_matrix = np.full((m, m), trans_prob)
    np.fill_diagonal(trans_matrix, stick_prob)
    return trans_matrix



def unif_cat_prior(m: int) -> np.ndarray:
    return np.full(m, 1.0 / m)


def imm_batch(models: List[AffineModel], m_trans: np.ndarray, m_prior: np.ndarray, mu_prior: np.ndarray,
              cov_prior: np.ndarray, zs: np.ndarray):
    m_ps = m_prior
    current_mus = mu_prior
    current_covs = cov_prior
    fused_mus = []
    fused_covs = []
    model_prob_time = []

    for i, z in enumerate(zs):
        m_ps, fused_mu, fused_cov, current_mus, current_covs = imm_kalman_filter(models, m_trans, m_ps, current_mus,
                                                                                 current_covs, z)
        fused_mus.append(fused_mu)
        fused_covs.append(fused_cov)
        model_prob_time.append(m_ps)

    return np.array(fused_mus), np.array(fused_covs), np.array(model_prob_time)


def imm_kalman_filter(models: List[AffineModel],
                      m_trans_ps: np.ndarray,
                      old_model_ps: np.ndarray,
                      old_mus: np.ndarray,
                      old_covs: np.ndarray,
                      z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Evaluate model mixing probabilities
    mix_probs = eval_mix_probs(models, m_trans_ps, old_model_ps)
    # print("Mix probs: ", mix_probs)

    # Evaluate mixing estimates and covariances
    mix_mus = [sum([old_mus[j] * mix_probs[j][i] for j in range(len(models))]) for i in range(len(models))]
    mix_covs = [mixing_covariance(mix_probs[:][i], old_mus[i], old_covs, mix_mus[i]) for i in range(len(models))]

    # Run Kalman Filters for Each Model Est
    model_kf_ests = [kalman_filter(m, mix_mus[i], mix_covs[i], z) for i, m in enumerate(models)]

    # Update the probability of each model
    log_model_ps = []
    for i, kf_est in enumerate(model_kf_ests):
        log_mix_norm = np.log(sum([m_trans_ps[l][i] * old_model_ps[l] for l in range(len(models))]))
        log_res_pdf = multivariate_normal.logpdf(kf_est.residual, mean=np.zeros(kf_est.residual.shape), cov=kf_est.S)
        # if res_pdf == 0:
        #     print("Cannot be zero probability...")
        log_model_ps.append(log_mix_norm + log_res_pdf)

    the_total = logsumexp(log_model_ps)

    model_ps = np.exp(np.array(log_model_ps) - logsumexp(log_model_ps))

    # Compute the fused estimates
    fused_mu = sum([k_res.est_mu * model_p for k_res, model_p in zip(model_kf_ests, model_ps)])
    fused_cov = sum(
        [(k_res.est_cov + np.diag(fused_mu - k_res.est_mu) @ np.diag(fused_mu - k_res.est_mu).T) * model_p for
         k_res, model_p in
         zip(model_kf_ests, model_ps)])

    return model_ps, fused_mu, fused_cov, np.array([k_res.est_mu for k_res in model_kf_ests]), np.array(
        [k_res.est_cov for k_res in model_kf_ests])


def eval_mix_probs(models: List[AffineModel], m_trans_ps: np.ndarray, old_model_ps: np.ndarray) -> np.ndarray:
    mix_probs = np.zeros((len(models), len(models)))
    for i, j in product(range(len(models)), repeat=2):
        mix_norm = sum([m_trans_ps[l][i] * old_model_ps[l] for l in range(len(models))])
        mix_probs[j][i] = m_trans_ps[i][j] * old_model_ps[j] / mix_norm

    return mix_probs


def mixing_covariance(model_mix_ps: np.ndarray, prev_mu: np.ndarray, prev_covs: np.ndarray,
                      mix_mu: np.ndarray) -> np.ndarray:
    cov_sum = []
    for j in range(len(model_mix_ps)):
        mp_diff = mix_mu - prev_mu
        mpd_sq = np.diag(mp_diff) @ np.diag(mp_diff).T
        cov_sum.append((prev_covs[j] + mpd_sq) * model_mix_ps[j])
    return sum(cov_sum)


def kalman_filter_batch(env_model: AffineModel, mu_0, cov_0, zs) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_mus = []
    pred_covs = []

    est_mus = []
    est_covs = []

    est_mu = mu_0
    est_cov = cov_0
    for z in zs:
        k_res = kalman_filter(env_model, est_mu, est_cov, z)

        est_mu = k_res.est_mu
        est_cov = k_res.est_cov

        pred_mus.append(k_res.pred_mu)
        pred_covs.append(k_res.pred_cov)

        est_mus.append(k_res.est_mu)
        est_covs.append(k_res.est_cov)

    return np.array(pred_mus), np.array(pred_covs), np.array(est_mus), np.array(est_covs)


def kalman_filter(env_model: AffineModel,
                  old_mu,
                  old_cov: np.ndarray,
                  z: np.ndarray) -> KalmanResult:
    pred_mu, pred_cov = kalman_predict(env_model, old_mu, old_cov)
    k_res = kalman_update(pred_mu, pred_cov, env_model, z)

    return k_res


def kalman_predict(model: AffineModel, old_mu: np.ndarray, old_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred_mu = model.F @ old_mu + model.E
    pred_cov = model.F @ old_cov @ model.F.T + np.diag(model.G) @ model.proc_noise_cov @ np.diag(model.G).T
    return pred_mu, pred_cov


def kalman_update(pred_mu: np.ndarray, pred_cov: np.ndarray, model: AffineModel, z: np.ndarray) -> KalmanResult:
    residual = z - model.H @ pred_mu
    S = model.H @ pred_cov @ model.H.T + model.measure_noise_cov
    K = pred_cov @ model.H.T @ np.linalg.inv(S)

    est_mu = pred_mu + K @ residual
    est_cov = pred_cov - K @ S @ K.T

    return KalmanResult(pred_mu, pred_cov, est_mu, est_cov, residual, S)


def update_sim(x: np.ndarray, env_model: AffineModel) -> np.ndarray:
    noise = np.random.multivariate_normal(np.zeros(x.shape), env_model.proc_noise_cov)
    x_next = (env_model.F @ x) + (env_model.G.T @ noise) + env_model.E
    return x_next


def full_sim(x_0: np.ndarray, env_model: AffineModel, N: int) -> np.ndarray:
    xs = []
    x_next = x_0
    for i in range(N):
        x_next = update_sim(x_next, env_model)
        xs.append(x_next)
    xs = np.array(xs)
    return xs


if __name__ == "__main__":
    run()
