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
from matplotlib import pyplot as plt

from CarMPC import TaskConfig, RectObstacle, car_mpc, IntervalConstraint, CostWeights, receding_horizon
from anim_utils import animate_scenario


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
                             car_length=4.298,
                             car_width=1.674,
                             v_goal=31.29,  # == 70mph
                             v_max=45.8,
                             acc_max=11.5,
                             ang_vel_max=0.4,
                             lanes=lane_centres,
                             lane_targets=[],
                             collision_field_slope=0.80)

    start_state = InitialState(position=np.array([10.0, lane_centres[0]]), velocity=0.0, orientation=0, time_step=0)

    cws = CostWeights(x_prog=0.01, y_prog=0.1, jerk=1, v_track=2, lane_align=1, road_align=1, collision_pot=100, faster_left=0, braking=10)
    # cws = CostWeights(x_prog=0.01, y_prog=0.1, jerk=50, v_track=300, lane_align=1, road_align=10, collision_pot=0, faster_left=0, braking=0)

    dn_state_list = receding_horizon(end_time, 1.0, start_state, scenario, task_config, cws)

    dyn_obs_shape = Rectangle(width=task_config.car_width, length=task_config.car_length)
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

    animate_scenario(scenario, int(end_time / task_config.dt),
                     ego_v=dyn_obs, show=True)  # , save_path="complexAnim.gif")


    scenario_save_path = "scenarios/Complex_Solution.xml"
    fw = CommonRoadFileWriter(scenario, planning_problem_set, "Craig Innes", "University of Edinburgh")
    fw.write_to_file(scenario_save_path, OverwriteExistingFile.ALWAYS)


if __name__ == "__main__":
    run()
