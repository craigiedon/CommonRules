import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType, StaticObstacle
from commonroad.scenario.state import InitialState, KSState
from commonroad.scenario.trajectory import State, Trajectory

from CarMPC import TaskConfig, RectObstacle, car_mpc, IntervalConstraint, CostWeights
from utils import animate_scenario


def run():
    file_path = "scenarios/Complex.xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()


    # Ford Escort Config. See Commonroad Vehicle Model Documentation
    lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [6, 10]]

    goal_state = planning_problem_set.find_planning_problem_by_id(1).goal.state_list[0].position.center
    start_time = 0.0
    end_time = 6.0
    task_config = TaskConfig(dt=0.1,
                             x_goal=goal_state[0],
                             y_goal=goal_state[1],
                             y_bounds=(0.0, 7.0),
                             car_width=4.298,
                             car_height=1.674,
                             v_goal=31.29, # == 70mph
                             # v_goal= 20.352,
                             v_max=45.8,
                             # v_max=10,
                             acc_max=11.5,
                             ang_vel_max=0.4,
                             lanes=lane_centres,
                             lane_targets=[],
                             collision_field_slope=0.9)
    # lane_targets=[IntervalConstraint(0, (0.0, 3.4)), IntervalConstraint(1, (3.5, 8.0))])

    start_state = InitialState(position=np.array([10.0, lane_centres[0]]), velocity=0.0, orientation=0, time_step=0)
    # obstacles = [
    #     RectObstacle(40.0, 4.5, 4.3, 1.674, 0.0),
    #     RectObstacle(5.0, 4.5, 4.3, 1.674, 0.0)
    # ]
    static_obstacles = scenario.static_obstacles
    dynamic_obstacles = scenario.dynamic_obstacles

    cws = CostWeights(x_prog=0.0001, y_prog=0.0, jerk=5, v_track=10, lane_align=1, collision_pot=20)
    empty_costs = CostWeights(0,0,0,0,0,0,0,0)

    current_time = start_time
    current_state = start_state
    dn_state_list = []
    i = 1

    # free_res = car_mpc(current_time, current_time + 1, current_state, task_config, [], [], cws)
    # res = car_mpc(current_time, current_time + 1, current_state, task_config, static_obstacles, dynamic_obstacles, cws, free_res)
    res = None
    horizon_length = 1.0
    while current_time < end_time:
        print(f"i: {i} ###################################")
        # free_res = car_mpc(current_time, current_time + horizon_length, current_state, task_config, [], [], cws, res)
        res = car_mpc(current_time, current_time + horizon_length, current_state, task_config, static_obstacles, dynamic_obstacles, cws, res)
        current_state = KSState(position=np.array([res.xs[1], res.ys[1]]), velocity=res.vs[1], orientation=res.hs[1], time_step=i)
        dn_state_list.append(current_state)
        current_time += task_config.dt
        i += 1

    print(len(dn_state_list))


    dyn_obs_shape = Rectangle(width=task_config.car_height, length=task_config.car_width)
    # dyn_obs_init = InitialState(position=[start_state.x, start_state.y], velocity=start_state.v, orientation=start_state.heading, time_step=0)
    # dn_state_list = [KSState(position=[x,y], velocity=v, orientation=h, time_step=i) for i, (x, y, v, h) in enumerate(zip(res.xs, res.ys, res.vs, res.hs))][1:]
    dyn_obs_traj = Trajectory(1, dn_state_list)
    dyn_obs_pred = TrajectoryPrediction(dyn_obs_traj, dyn_obs_shape)
    dyn_obs = DynamicObstacle(scenario.generate_object_id(),
                              ObstacleType.CAR,
                              dyn_obs_shape,
                              start_state,
                              dyn_obs_pred)


    scenario.add_objects(dyn_obs)

    animate_scenario(scenario, planning_problem_set, int((end_time - start_time) / task_config.dt), ego_v=dyn_obs)#save_path="cutInAnim.gif")

    # scenario_save_path = "scenarios/Complex.xml"
    # fw = CommonRoadFileWriter(scenario, planning_problem_set, "Craig Innes", "University of Edinburgh")
    # fw.write_to_file(scenario_save_path, OverwriteExistingFile.ALWAYS)

if __name__ == "__main__":
    run()
