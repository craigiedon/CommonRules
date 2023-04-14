from dataclasses import dataclass
from itertools import product
from typing import Tuple, List, Any

import numpy as np
from numpy.random import multivariate_normal as rand_mvn
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, IMMEstimator
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
    p_ref: float


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


def lat_model(ts: float, k_d: float, k_p: float, p_ref: float, q_p: float, q_m: float) -> StateFeedbackModel:
    return StateFeedbackModel(
        ts=ts,
        F=np.array([
            [1.0, ts],
            [-ts * k_p, 1 - ts * k_d],
        ]),
        G=np.array([(ts ** 2.0) / 2.0, ts]),
        E=np.array([0.0,
                    ts * k_p * p_ref]),
        proc_noise_cov=np.identity(2) * q_p,
        H=np.array([[1.0, 0.0]]),
        measure_noise_cov=np.identity(1) * q_m,
        k_p=k_p,
        k_d=k_d,
        p_ref=p_ref
    )


def kf_to_filterpy(model: AffineModel, x_0: np.ndarray) -> KalmanFilter:
    f = KalmanFilter(dim_x=len(x_0), dim_z=len(model.H @ x_0), dim_u=len(model.E))
    f.x = x_0
    f.F = model.F

    # Hack for Affine shift --- Put in the control?
    f.B = np.diag(model.E)

    f.H = model.H
    f.Q = np.diag(model.G) @ model.proc_noise_cov @ np.diag(model.G.T)
    f.R = model.measure_noise_cov

    return f


@dataclass
class DiagFilterConfig:
    N: int  # Number of timesteps
    dt: float  # Timestep duration
    x_0: np.ndarray  # Starting state
    proc_noise: float
    measure_noise: float


def run():
    # Longitudinal Models
    N = 200
    ts = 0.05

    # Long Simulation
    long_config = DiagFilterConfig(N=200, dt=0.05, x_0=np.array([0.0, 10.0, 5.0]), proc_noise=1.0, measure_noise=1.0)
    long_models = [c_vel_long_model(long_config.dt, long_config.proc_noise, long_config.measure_noise),
                   c_acc_long_model(long_config.dt, long_config.proc_noise, long_config.measure_noise)]
    long_sim = c_acc_long_model(long_config.dt, 0.0, long_config.measure_noise)

    long_accs = np.zeros(N)
    long_accs[0: N // 2] = 5.0
    long_accs[N // 2:] = -40
    xs_long, zs_long = car_sim_long(long_config.x_0[0], long_config.x_0[1], long_accs, long_sim, N)

    # IMM Estimation
    long_mus, long_covs, mps_long = imm_batch(long_models, np.array([[0.5, 0.5], [0.5, 0.5]]),
                                              np.array([0.5, 0.5]),
                                              np.tile(long_config.x_0, (len(long_models), 1)),
                                              np.tile(np.identity(3), (len(long_models), 1, 1)), zs_long)

    # Lat Models
    lane_positions = [-3.0, 0.0, 3.0]
    lat_config = DiagFilterConfig(N=200, dt=0.05, x_0=np.array([lane_positions[0], 0.0]), proc_noise=1.0,
                                  measure_noise=0.1)
    lat_models: List[StateFeedbackModel] = [
        lat_model(ts, kd, 4.0, p_ref, lat_config.proc_noise, lat_config.measure_noise) for kd in
        np.linspace(3.0, 5.0, 3)
        for p_ref in lane_positions]

    # Lat Simulation
    # Start in right lane, then at 3 seconds, lange change into middle
    lane_durations = [(3.0, 0), (6.0, 1), (1.0, 2)]
    xs_lat, zs_lat = car_sim_lat(lat_config.x_0, lane_durations, lane_positions, lat_config)

    lat_mus, lat_covs, mps_lat = imm_batch(lat_models,
                                           sticky_m_trans(len(lat_models), 0.95),
                                           unif_cat_prior(len(lat_models)),
                                           np.tile(lat_config.x_0, (len(lat_models), 1)),
                                           np.tile(np.identity(len(lat_config.x_0)), (len(lat_models), 1, 1)), zs_lat)

    assert len(lat_mus) == len(zs_lat)

    fig, axs = plt.subplots(1, 1)
    axs.scatter(ts * np.arange(len(xs_lat)), xs_lat[:, 0], color='black', facecolors='none', s=4.0)
    axs.scatter(ts * np.arange(len(zs_lat)), zs_lat, marker='x')

    axs.plot(ts * np.arange(len(lat_mus)), lat_mus[:, 0], color='orange')

    pred_ts = np.arange(0, 9)
    for pt in pred_ts:
        pts = int(pt / lat_config.dt)
        prediction_at_t = target_state_prediction(lat_mus[pts], lat_models, mps_lat[pts], N - pts)
        axs.plot(ts * np.arange(pts, N), prediction_at_t[:, 0], color='purple', alpha=0.5)

    # for fp_lat_trace in fpy_mus_lat:
    for single_model in lat_models:
        single_pred_mus, single_pred_covs, single_est_mus, single_est_covs = kalman_filter_batch(single_model,
                                                                                                 lat_config.x_0,
                                                                                                 np.identity(
                                                                                                     len(lat_config.x_0)),
                                                                                                 zs_lat)
        axs.plot(ts * np.arange(len(single_est_mus)), single_est_mus[:, 0], color='pink', alpha=0.5)

    axs.plot(ts * np.arange(len(xs_lat)), np.repeat(0.0, len(xs_lat)), linestyle='--', color='black', alpha=0.2)
    axs.plot(ts * np.arange(len(xs_lat)), np.repeat(3.0, len(xs_lat)), linestyle='--', color='black', alpha=0.2)
    axs.plot(ts * np.arange(len(xs_lat)), np.repeat(-3.0, len(xs_lat)), linestyle='--', color='black', alpha=0.2)
    axs.set_xlabel("Time (s)")
    axs.set_ylabel("Latitude (m)")
    axs.set_ylim(-5, 5)

    plt.show()

    fig, axs = plt.subplots(1, 1)
    axs.scatter(xs_long[:, 0], xs_lat[:, 0], color='black', facecolors='none', s=4.0, label="True State")
    axs.scatter(zs_long[:, 0], zs_lat, marker='x', label="Sensor Observations")
    axs.plot(long_mus[:, 0], lat_mus[:, 0], color='orange', label="IMM")
    axs.set_xlabel("Longitude (m)")
    axs.set_ylabel("Latitude (m)")
    axs.legend(loc="best")
    plt.show()


def car_sim_long(p_0: float, vel_0: float, accelerations: np.ndarray, simulation_model: AffineModel, N: int) -> Tuple[
    np.ndarray, np.ndarray]:
    xs_long = []

    x_long_next = np.array([p_0, vel_0, accelerations[0]])

    for i in range(1, N):
        x_long_next = update_sim(x_long_next, simulation_model)
        x_long_next[2] = accelerations[i]
        xs_long.append(x_long_next)

    xs_long.append(update_sim(x_long_next, simulation_model))

    xs_long = np.array(xs_long)
    zs_long = np.array([measure_from_state(x, simulation_model) for x in xs_long])

    assert len(xs_long) == len(zs_long) == N

    return xs_long, zs_long


def car_sim_lat(x_lat_0: np.ndarray, lane_durations: List[Tuple[float, int]], lane_positions: List[float],
                lat_config: DiagFilterConfig) -> Tuple[np.ndarray, np.ndarray]:
    lat_lane_models = [lat_model(lat_config.dt, 4.0, 4.0, lp, lat_config.proc_noise, lat_config.measure_noise) for lp in
                       lane_positions]
    xs_lat = []
    x_lat_next = x_lat_0

    duration_total = np.sum([d for d, _ in lane_durations])

    for duration, lane_idx in lane_durations:
        xs_lat.extend(
            full_sim(x_lat_next, lat_lane_models[lane_idx], int(duration / lat_config.dt)))
        x_lat_next = xs_lat[-1]

    xs_lat = np.array(xs_lat)
    zs_lat = np.array([measure_from_state(x, lat_lane_models[0]) for x in xs_lat])

    assert len(xs_lat) == len(zs_lat) == lat_config.N

    return xs_lat, zs_lat


def target_state_prediction(x_est: np.ndarray, models: List[AffineModel], model_probs: List[float],
                            prediction_steps: int) -> np.ndarray:
    top_model = models[np.argmax(model_probs)]

    x_current = x_est
    predictions = []
    for i in range(prediction_steps):
        x_current = update_sim_noiseless(x_current, top_model)
        predictions.append(x_current)
    predictions = np.array(predictions)

    return predictions


def all_model_predictions(x_est: np.ndarray, models: List[AffineModel], prediction_steps: int) -> List[np.ndarray]:
    model_preds = []
    for model in models:
        x_current = x_est
        predictions = []
        for i in range(prediction_steps):
            x_current = update_sim_noiseless(x_current, model)
            predictions.append(x_current)
        model_preds.append(np.array(predictions))

    return model_preds


def measurements_from_traj(xs: np.ndarray, sim_model: AffineModel) -> np.ndarray:
    return np.array([measure_from_state(x, sim_model) for x in xs])


def measure_from_state(x: np.ndarray, sim_model: AffineModel) -> np.ndarray:
    n_dims = len(sim_model.H)

    proj_obs = sim_model.H @ x
    measure_noise = rand_mvn(np.zeros(n_dims), sim_model.measure_noise_cov)

    return proj_obs + measure_noise


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


def closest_lane_prior(lat_start: float, lat_models: List[StateFeedbackModel], match_val: float) -> np.ndarray:
    assert 0.0 < match_val < 1.0

    lat_prior = []
    for lm in lat_models:
        if np.abs(lm.p_ref - lat_start) < 0.05:
            lat_prior.append(match_val)
        else:
            lat_prior.append(1.0 - match_val)
    lat_prior = np.array(lat_prior) / np.sum(lat_prior)
    return lat_prior


def imm_batch(models: List[AffineModel], m_trans: np.ndarray, m_prior: np.ndarray, mu_prior: np.ndarray,
              cov_prior: np.ndarray, zs: np.ndarray):

    imm_res = IMMResult(m_prior, None, None, mu_prior, cov_prior)
    fused_mus = []
    fused_covs = []
    model_prob_time = []

    for i, z in enumerate(zs):
        imm_res = imm_kalman_filter(models, m_trans, imm_res.model_ps, imm_res.model_mus, imm_res.model_covs, z)
        fused_mus.append(imm_res.fused_mu)
        fused_covs.append(imm_res.fused_cov)
        model_prob_time.append(imm_res.model_ps)

    return np.array(fused_mus), np.array(fused_covs), np.array(model_prob_time)

@dataclass
class IMMResult:
    model_ps: np.ndarray
    fused_mu: np.ndarray
    fused_cov: np.ndarray
    model_mus: np.ndarray
    model_covs: np.ndarray


def imm_kalman_filter(models: List[AffineModel],
                      m_trans_ps: np.ndarray,
                      old_model_ps: np.ndarray,
                      old_mus: np.ndarray,
                      old_covs: np.ndarray,
                      z: np.ndarray) -> IMMResult:
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
    fused_mu = np.sum([k_res.est_mu * model_p for k_res, model_p in zip(model_kf_ests, model_ps)], axis=0)
    fused_cov = np.sum(
        [(k_res.est_cov + np.diag(fused_mu - k_res.est_mu) @ np.diag(fused_mu - k_res.est_mu).T) * model_p for
         k_res, model_p in
         zip(model_kf_ests, model_ps)], axis=0)

    return IMMResult(model_ps, fused_mu, fused_cov, np.array([k_res.est_mu for k_res in model_kf_ests]), np.array(
        [k_res.est_cov for k_res in model_kf_ests]))


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


def update_sim_noiseless(x: np.ndarray, env_model: AffineModel) -> np.ndarray:
    x_next = (env_model.F @ x) + env_model.E
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
