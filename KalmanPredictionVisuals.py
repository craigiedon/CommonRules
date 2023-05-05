from collections import defaultdict
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.mp_renderer import MPRenderer
from matplotlib import animation
from matplotlib.animation import FuncAnimation

from CarMPC import RecedingHorizonStats
from immFilter import measure_from_state, c_acc_long_model, DiagFilterConfig, lat_model, t_state_pred, \
    imm_batch, sticky_m_trans, unif_cat_prior, c_vel_long_model, closest_lane_prior, all_model_predictions
from trafficRules import rot_mat


def run():
    file_path = "scenarios/Complex_Solution.xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

    # Get the dynamic obstacles
    T = 60
    dt = 0.1

    # Get out their "predictions" (Really this is just their true position over time)
    state_dict_traj = [scenario.obstacle_states_at_time_step(i) for i in range(T)]

    # Long Simulation
    long_models = [
        c_vel_long_model(dt, 1.0, 0.1),
        c_acc_long_model(dt, 1.0, 0.1)
    ]

    lane_centres = [-1.75, 1.75, 5.25]

    # Lat Simulation
    lat_models = [
        lat_model(dt, kd, 7, p_ref, 0.1, 0.1)
        for kd in np.linspace(3.0, 5.0, 3)
        for p_ref in lane_centres]

    # Convert to kalman filter format? (long_pos, x_long_vel, long_acc)
    # xs_long_dict_traj = []
    # xs_lat_dict_traj = []
    obs_tru_long = defaultdict(list)
    obs_tru_lat = defaultdict(list)

    obs_zs_long = defaultdict(list)
    obs_zs_lat = defaultdict(list)

    for sd in state_dict_traj:
        for obj_id, s in sd.items():
            print("Obj: ", obj_id)
            long_pos, lat_pos = s.position
            r = rot_mat(s.orientation)

            long_vel, lat_vel = r @ np.array([s.velocity, 0.0])
            long_acc, _ = r @ np.array([s.acceleration, 0.0])

            long_state = np.array([long_pos, long_vel, long_acc])
            lat_state = np.array([lat_pos, lat_vel])

            obs_tru_long[obj_id].append(long_state)
            obs_tru_lat[obj_id].append(lat_state)

            # At each time step, add some noise to these positions
            z_long = measure_from_state(long_state, long_models[0])
            z_lat = measure_from_state(lat_state, lat_models[0])

            obs_zs_long[obj_id].append(z_long)
            obs_zs_lat[obj_id].append(z_lat)

    # The Kalman filter bit: I want all the mu estimates at each bit (should probably be a one liner?)
    imm_long_stats = {}
    imm_lat_stats = {}

    for obs in scenario.obstacles:
        x_longs = np.array(obs_tru_long[obs.obstacle_id])
        x_lats = np.array(obs_tru_lat[obs.obstacle_id])
        z_longs = np.array(obs_zs_long[obs.obstacle_id])
        z_lats = np.array(obs_zs_lat[obs.obstacle_id])

        imm_long_stats[obs.obstacle_id] = imm_batch(long_models, sticky_m_trans(len(long_models), 0.95),
                                                    unif_cat_prior(len(long_models)),
                                                    np.tile(x_longs[0], (len(long_models), 1)),
                                                    np.tile(np.identity(3), (len(long_models), 1, 1)), z_longs)

        imm_lat_stats[obs.obstacle_id] = imm_batch(lat_models,
                                                   sticky_m_trans(len(lat_models), 0.95),
                                                   closest_lane_prior(x_lats[0, 0], lat_models, 0.9),
                                                   np.tile(x_lats[0], (len(lat_models), 1)),
                                                   np.tile(np.identity(len(x_lats[0])), (len(lat_models), 1, 1)),
                                                   z_lats)

    obstacle_long_mus: Dict[int, np.ndarray] = {o.obstacle_id: imm_long_stats[o.obstacle_id][0] for o in
                                                scenario.obstacles}
    obstacle_lat_mus: Dict[int, np.ndarray] = {o.obstacle_id: imm_lat_stats[o.obstacle_id][0] for o in
                                               scenario.obstacles}

    obstacle_long_preds: Dict[int, np.ndarray] = {}
    obstacle_lat_preds: Dict[int, np.ndarray] = {}

    for o in scenario.obstacles:
        long_mu = obstacle_long_mus[o.obstacle_id]
        lat_mu = obstacle_lat_mus[o.obstacle_id]

        obstacle_long_preds[o.obstacle_id] = np.stack(
            [t_state_pred(long_mu[i], long_models, imm_long_stats[o.obstacle_id][2][i], 15) for i in
             range(T)])
        obstacle_lat_preds[o.obstacle_id] = np.stack(
            [t_state_pred(lat_mu[i], lat_models, imm_lat_stats[o.obstacle_id][2][i], 15) for i in range(T)])

    # Mark the "noisy" steps on the visualizer
    fig, ax = plt.subplots(1, 1, figsize=(25, 2.5))

    rnd = MPRenderer(ax=ax)
    rnd.draw_params.dynamic_obstacle.trajectory.draw_continuous = True
    rnd.draw_params.dynamic_obstacle.show_label = True

    ani = FuncAnimation(fig,
                        lambda i: animate_kalman_predictions(i, ax, rnd, scenario, obs_tru_long, obs_tru_lat,
                                                             obstacle_long_mus, obstacle_lat_mus, obstacle_long_preds,
                                                             obstacle_lat_preds,
                                                             obs_zs_long, obs_zs_lat, False, True),
                        frames=T, interval=30, repeat=True, repeat_delay=200)
    # ani.save("kalmanPredictions.gif", animation.PillowWriter(fps=3))

    ax.set_xlim(0, 250)
    ax.set_ylim(-5, 5)

    plt.tight_layout()
    plt.show()


def animate_kalman_predictions(i, ax, rnd: MPRenderer, scenario: Scenario,
                               p_stats: Dict[int, RecedingHorizonStats],
                               plot_tru_pos: bool = False,
                               plot_obs_pos: bool = False):
    rnd.draw_params.time_begin = i
    rnd.draw_params.time_end = i + 15
    scenario.draw(rnd)
    rnd.render()

    for o_id, ps in p_stats.items():
        x_longs = np.array(ps.true_longs)
        x_lats = np.array(ps.true_lats)
        z_longs = np.array(ps.observed_longs)
        z_lats = np.array(ps.observed_lats)

        if plot_tru_pos:
            ax.plot(x_longs[:i, 0], x_lats[:i, 0], zorder=100)

        if plot_obs_pos:
            detected_longs = [zl[0] for zl in z_longs[:i] if zl is not None]
            detected_lats = [zl for zl in z_lats[:i] if zl is not None]
            ax.scatter(detected_longs, detected_lats, zorder=100, s=25, marker='x', alpha=1.0)

        ax.plot(ps.prediction_traj_longs[i][:, 0],
                ps.prediction_traj_lats[i][:, 0],
                color='purple', alpha=0.5, zorder=1000)

        ax.scatter(ps.est_longs[i][0], ps.est_lats[i][0], color='purple', zorder=1000)


if __name__ == "__main__":
    run()
