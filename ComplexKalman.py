from typing import List

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType, StaticObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import InitialState, KSState
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.visualization.mp_renderer import MPRenderer
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from CarMPC import TaskConfig, RectObstacle, car_mpc, IntervalConstraint, CostWeights, receding_horizon, \
    kalman_receding_horizon
from KalmanPredictionVisuals import animate_kalman_predictions
from immFilter import c_vel_long_model, c_acc_long_model, lat_model
from utils import animate_scenario


def run():
    file_path = "scenarios/Complex.xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

    # Ford Escort Config. See Commonroad Vehicle Model Documentation
    lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [6, 10]]

    goal_state = planning_problem_set.find_planning_problem_by_id(1).goal.state_list[0].position.center
    end_time = 6.0
    task_config = TaskConfig(dt=0.1,
                             x_goal=goal_state[0],
                             y_goal=goal_state[1],
                             y_bounds=(0.0, 7.0),
                             car_width=4.298,
                             car_height=1.674,
                             v_goal=31.29,  # == 70mph
                             v_max=45.8,
                             acc_max=11.5,
                             ang_vel_max=0.4,
                             lanes=lane_centres,
                             lane_targets=[],
                             collision_field_slope=0.80)

    start_state = InitialState(position=np.array([10.0, lane_centres[0]]), velocity=0.0, orientation=0, time_step=0)

    cws = CostWeights(x_prog=0.01, y_prog=0.1, jerk=1, v_track=2, lane_align=1, road_align=1, collision_pot=100,
                      faster_left=1.0, braking=10)

    long_models = [
        c_vel_long_model(task_config.dt, 1.0, 0.1),
        c_acc_long_model(task_config.dt, 1.0, 0.1)
    ]

    lat_models = [
        lat_model(task_config.dt, kd, 7, p_ref, 0.1, 0.1)
        for kd in np.linspace(3.0, 5.0, 3)
        for p_ref in lane_centres]

    dn_state_list = kalman_receding_horizon(end_time, 2.5, start_state, scenario, task_config, long_models, lat_models,
                                            cws)

    dyn_obs_shape = Rectangle(width=task_config.car_height, length=task_config.car_width)
    dyn_obs_traj = Trajectory(1, dn_state_list)
    dyn_obs_pred = TrajectoryPrediction(dyn_obs_traj, dyn_obs_shape)
    ego_id = 100
    dyn_obs = DynamicObstacle(ego_id,
                              ObstacleType.CAR,
                              dyn_obs_shape,
                              start_state,
                              dyn_obs_pred)

    scenario.add_objects(dyn_obs)

    # plt.plot([s.acceleration for s in dn_state_list])
    # plt.show()

    # Show a visual that has the prediction parts too
    fig, ax = plt.subplots(figsize=(25, 3))
    rnd = MPRenderer(ax=ax)
    rnd.draw_params.dynamic_obstacle.trajectory.draw_continuous = True
    rnd.draw_params.dynamic_obstacle.show_label = True
    ani = FuncAnimation(fig, lambda i: animate_kalman_predictions(i, ax, rnd, scenario, obs_tru_long, obs_tru_lat, obs_long_mus, obs_lat_mus, obs_long_preds, obs_lat_preds,
                                                                  obs_zs_long, obs_zs_lat, False, True),
                        frames=len(dn_state_list), interval=30, repeat=True, repeat_delay=200)
    plt.tight_layout()
    plt.show()

    animate_scenario(scenario, int(end_time / task_config.dt),
                     ego_v=dyn_obs, show=True)  # , save_path="complexAnim.gif")

    # scenario_save_path = "scenarios/Complex_Solution.xml"
    # fw = CommonRoadFileWriter(scenario, planning_problem_set, "Craig Innes", "University of Edinburgh")
    # fw.write_to_file(scenario_save_path, OverwriteExistingFile.ALWAYS)


if __name__ == "__main__":
    run()
