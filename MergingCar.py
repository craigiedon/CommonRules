import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType, StaticObstacle
from commonroad.scenario.state import InitialState, KSState, CustomState
from commonroad.scenario.trajectory import State, Trajectory

from CarMPC import RectObstacle, car_mpc
from TaskConfig import TaskConfig, IntervalConstraint, CostWeights
from anim_utils import animate_scenario


def run():
    file_path = "scenarios/CutIn.xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()


    # Ford Escort Config. See Commonroad Vehicle Model Documentation
    lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [3, 6]]
    goal_state = planning_problem_set.find_planning_problem_by_id(1).goal.state_list[0].position.center
    task_config = TaskConfig(dt=0.1,
                             x_goal=goal_state[0],
                             y_goal=lane_centres[1],
                             y_bounds=(-3.5, 7.0),
                             car_length=4.298,
                             car_width=1.674,
                             v_goal=22.29,  # == 70mph
                             # v_goal= 20.352,
                             v_max=45.8,
                             acc_max=11.5,
                             ang_vel_max=0.4,
                             lanes=lane_centres,
                             collision_field_slope=1.0,
                             # lane_targets=[])
                             lane_targets=[IntervalConstraint(0, (0.0, 1.5)), IntervalConstraint(1, (1.6, 8.0))])

    start_state = InitialState(position=[20.0, lane_centres[0]], velocity=0.0, orientation=0, time_step=0)

    start_time = 0.0
    end_time = 8.0
    res = car_mpc(start_time, end_time, start_state, task_config, scenario.static_obstacles, scenario.dynamic_obstacles, CostWeights(x_prog=0.0001, y_prog=0.0, v_track=20, acc=5, lane_align=5, collision_pot=1000))

    dyn_obs_shape = Rectangle(width=1.674, length=4.298)

    dn_state_list = [CustomState(position=[x,y], velocity=v, acceleration=a, orientation=h, time_step=i) for i, (x, y, v, a, h) in enumerate(zip(res.xs, res.ys, res.vs, res.accs, res.hs))][1:]

    dyn_obs_traj = Trajectory(1, dn_state_list)
    dyn_obs_pred = TrajectoryPrediction(dyn_obs_traj, dyn_obs_shape)
    dyn_obs = DynamicObstacle(scenario.generate_object_id(),
                              ObstacleType.CAR,
                              dyn_obs_shape,
                              start_state,
                              dyn_obs_pred)


    scenario.add_objects(dyn_obs)

    animate_scenario(scenario, int(end_time / task_config.dt), show=True) #save_path="cutInAnim.gif")

    scenario_save_path = "scenarios/Complex.xml"
    fw = CommonRoadFileWriter(scenario, planning_problem_set, "Craig Innes", "University of Edinburgh")
    fw.write_to_file(scenario_save_path, OverwriteExistingFile.ALWAYS)

if __name__ == "__main__":
    run()
