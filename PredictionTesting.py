import matplotlib.pyplot as plt
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from matplotlib.animation import FuncAnimation

from immFilter import measure_from_state, c_acc_long_model, DiagFilterConfig, lat_model, target_state_prediction, \
    imm_batch, sticky_m_trans, unif_cat_prior, c_vel_long_model
from trafficRules import rot_mat

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
    lat_model(dt, kd, 4.0, p_ref, 1.0, 0.1) for kd in
    np.linspace(3.0, 5.0, 3)
    for p_ref in lane_centres]

# Convert to kalman filter format? (long_pos, x_long_vel, long_acc)
xs_long_dict_traj = []
xs_lat_dict_traj = []

zs_long_dict_traj = []
zs_lat_dict_traj = []
for sd in state_dict_traj:
    x_long_dict = {}
    x_lat_dict = {}

    z_long_dict = {}
    z_lat_dict = {}

    for obj_id, s in sd.items():
        print("Obj: ", obj_id)
        long_pos, lat_pos = s.position
        r = rot_mat(s.orientation)

        long_vel, lat_vel = r @ np.array([s.velocity, 0.0])
        long_acc, _ = r @ np.array([s.acceleration, 0.0])

        # TODO: There is going to be a problem here with time-step granularities (are velocities in m/s? Or relative to current dt?)
        long_state = np.array([long_pos, long_vel, long_acc])
        lat_state = np.array([lat_pos, lat_vel])

        x_long_dict[obj_id] = long_state
        x_lat_dict[obj_id] = lat_state

        # At each time step, add some noise to these positions
        z_long = measure_from_state(long_state, long_models[0])
        z_lat = measure_from_state(lat_state, lat_models[0])

        z_long_dict[obj_id] = z_long
        z_lat_dict[obj_id] = z_lat

    xs_long_dict_traj.append(x_long_dict)
    xs_lat_dict_traj.append(x_lat_dict)
    zs_long_dict_traj.append(z_long_dict)
    zs_lat_dict_traj.append(z_lat_dict)

# The Kalman filter bit: I want all the mu estimates at each bit (should probably be a one liner?)
imm_long_stats = {}
imm_lat_stats = {}

for obs in scenario.obstacles:
    x_longs = np.array([sd[obs.obstacle_id] for sd in xs_long_dict_traj])
    x_lats = np.array([sd[obs.obstacle_id] for sd in xs_lat_dict_traj])
    z_longs = np.array([sd[obs.obstacle_id] for sd in zs_long_dict_traj])
    z_lats = np.array([sd[obs.obstacle_id] for sd in zs_lat_dict_traj])

    # TODO: This probably wont work because of the dt Timesteps thing!
    imm_long_stats[obs.obstacle_id] = imm_batch(long_models, np.array([[0.5, 0.5], [0.5, 0.5]]),
                                                np.array([0.5, 0.5]),
                                                np.tile(x_longs[0], (len(long_models), 1)),
                                                np.tile(np.identity(3), (len(long_models), 1, 1)), z_longs)

    imm_lat_stats[obs.obstacle_id] = imm_batch(lat_models,
                                               sticky_m_trans(len(lat_models), 0.95),
                                               unif_cat_prior(len(lat_models)),
                                               np.tile(x_lats[0], (len(lat_models), 1)),
                                               np.tile(np.identity(len(x_lats[0])), (len(lat_models), 1, 1)), z_lats)

# Mark the "noisy" steps on the visualizer
fig, ax = plt.subplots(1, 1, figsize=(25, 2.5))

rnd = MPRenderer(ax=ax)
rnd.draw_params.dynamic_obstacle.trajectory.draw_continuous = True
rnd.draw_params.dynamic_obstacle.show_label = True


def animate(i):
    rnd.draw_params.time_begin = i
    rnd.draw_params.time_end = i + 15
    scenario.draw(rnd)
    rnd.render()

    prediction_steps = 15

    for obs in scenario.obstacles:
        x_longs = np.array([sd[obs.obstacle_id] for sd in xs_long_dict_traj])
        long_mus, long_covs, mps_long = imm_long_stats[obs.obstacle_id]

        x_long_predictions = target_state_prediction(long_mus[i], long_models, mps_long[i], prediction_steps)

        x_lats = np.array([sd[obs.obstacle_id] for sd in xs_lat_dict_traj])
        lat_mus, lat_covs, mps_lat = imm_lat_stats[obs.obstacle_id]

        x_lat_predictions = target_state_prediction(lat_mus[i], lat_models, mps_lat[i], prediction_steps)

        z_longs = np.array([sd[obs.obstacle_id] for sd in zs_long_dict_traj])
        z_lats = np.array([sd[obs.obstacle_id] for sd in zs_lat_dict_traj])

        # ax.plot(x_longs[max(0, i-5):i, 0], x_lats[max(0, i-5):i, 0], zorder=100)
        # ax.scatter(z_longs[max(0, i-5):i, 0], z_lats[max(0, i-5):i, 0], zorder=100, s=25, marker='x', alpha=1.0)

        # ax.plot(x_longs[:i, 0], x_lats[:i, 0], zorder=100)
        # ax.scatter(z_longs[:i, 0], z_lats[:i, 0], zorder=100, s=25, marker='x', alpha=1.0)

        # ax.plot(x_long_predictions[:, 0], x_lat_predictions[:, 0], color='purple', alpha=0.8, zorder=1000)
        ax.scatter(long_mus[i, 0], lat_mus[i, 0], color='purple', zorder=1000)


ani = FuncAnimation(fig, animate, frames=T, interval=32, repeat=True, repeat_delay=200)

plt.tight_layout()
plt.show()

# Run a kalman filter on the noisy measurements
# Log/Plot the kalman filter estimates on the graph too
# At each time step, predict the horizon (i.e., 1 second ahead)
# Try plotting / animating this?
