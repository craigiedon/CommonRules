import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType, StaticObstacle
from commonroad.scenario.trajectory import State, Trajectory

from CarMPC import TaskConfig, RectObstacle, car_mpc, CarState, IntervalConstraint
from utils import animate_scenario


def run():
    file_path = "scenarios/EmptyRamp.xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

    # stat_obs = StaticObstacle(scenario.generate_object_id(),
    #                           ObstacleType.CAR,
    #                           shape,
    #                           init_state)


    # Ford Escort Config. See Commonroad Vehicle Model Documentation
    lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [3, 6, 10]]
    goal_state = planning_problem_set.find_planning_problem_by_id(1).goal.state_list[0].position.center
    task_config = TaskConfig(time=8,
                             dt=0.1,
                             x_goal=goal_state[0],
                             y_goal=goal_state[1],
                             car_width=4.298,
                             car_height=1.674,
                             v_goal=31.29, # == 70mph
                             v_max=45.8,
                             acc_max=11.5,
                             ang_vel_max=0.4,
                             lanes=lane_centres,
                             lane_targets=[IntervalConstraint(2, (0.0, 2.0))])

    start_state = CarState(x=20.0, y=lane_centres[2], v=0.0, heading=0)
    # obstacles = [
    #     RectObstacle(40.0, 4.5, 4.3, 1.674, 0.0),
    #     RectObstacle(5.0, 4.5, 4.3, 1.674, 0.0)
    # ]
    obstacles = []

    res = car_mpc(start_state, task_config, obstacles)

    dyn_v = 31.29
    dyn_obs_shape = Rectangle(width=1.674, length=4.298)
    dyn_obs_init = State(position=[start_state.x, start_state.y], velocity=start_state.v, orientation=start_state.heading, time_step=0)

    dn_state_list = [State(position=[x, y], velocity=v, orientation=h, time_step=i) for i, (x, y, v, h) in enumerate(zip(res.xs, res.ys, res.vs, res.hs))][1:]
    # for i in range(1, timesteps + 1):
    #     new_pos = np.array([dyn_obs_init.position[0] + scenario.dt * i * dyn_v, dyn_obs_init.position[1]])
    #     dn_state_list.append(State(position=new_pos, velocity=dyn_v, orientation=0.02, time_step=i))

    dyn_obs_traj = Trajectory(1, dn_state_list)
    dyn_obs_pred = TrajectoryPrediction(dyn_obs_traj, dyn_obs_shape)
    dyn_obs = DynamicObstacle(scenario.generate_object_id(),
                              ObstacleType.CAR,
                              dyn_obs_shape,
                              dyn_obs_init,
                              dyn_obs_pred)

    scenario.add_objects(dyn_obs)
    animate_scenario(scenario, planning_problem_set, int(task_config.time / task_config.dt), save_path="cutInAnim.gif")
    # print(res)

    scenario_save_path = "scenarios/CutIn.xml"
    fw = CommonRoadFileWriter(scenario, planning_problem_set, "Craig Innes", "University of Edinburgh")
    fw.write_to_file(scenario_save_path, OverwriteExistingFile.ALWAYS)

if __name__ == "__main__":
    run()