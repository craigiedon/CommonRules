import copy
from typing import Optional

from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.mp_renderer import MPRenderer
from matplotlib import pyplot as plt, animation
from matplotlib.animation import FuncAnimation


def animate_scenario_old(scenario: Scenario, planning_problem_set, timesteps: int, save_path=None):
    fig, ax = plt.subplots(figsize=(25, 3))
    rnd = MPRenderer(ax=ax)
    dps = rnd.draw_params
    rnd.draw_params["planning_problem"]["initial_state"]["state"].update({"zorder": 20})
    is_s = rnd.draw_params["planning_problem"]["initial_state"]["state"]
    print(dps)
    def animate(i):
        # rnd.draw_params.time_begin = i
        scenario.draw(rnd, draw_params={"time_begin": i})
        # ego_vehicle.draw(rnd, draw_params={'time_begin': i, 'dynamic_obstacle': {
        #     'vehicle_shape': {'occupancy': {'shape': {'rectangle': {
        #         'facecolor': 'g'}}}}}})
        planning_problem_set.draw(rnd)
        rnd.render()

    ani = FuncAnimation(fig, animate, frames=timesteps, interval=32, repeat=True, repeat_delay=200)
    if save_path is not None:
        ani.save(save_path, animation.PillowWriter(fps=30))
    plt.show()

def animate_scenario(scenario: Scenario, planning_problem_set, timesteps: int, ego_v:Optional[DynamicObstacle]=None, save_path=None):
    fig, ax = plt.subplots(figsize=(25, 3))

    rnd = MPRenderer(ax=ax)
    dps = rnd.draw_params
    is_s = rnd.draw_params["planning_problem"]["initial_state"]["state"]
    rnd.draw_params.dynamic_obstacle.trajectory.draw_continuous = True
    # rnd.draw_params.dynamic_obstacle.trajectory.zorder = 10000


    # rnd.draw_params.dynamic_obstacle.trajectory.draw_trajectory = True
    # rnd.draw_params.dynamic_obstacle.trajectory.draw_continuous = False
    # rnd.draw_params.dynamic_obstacle.trajectory.line_width = 100.0 <- This is key to seeing bug - there is just something broken about the ellipse offsets
    # rnd.draw_params.dynamic_obstacle.trajectory.shape.draw_mesh = True
    # rnd.draw_params.dynamic_obstacle.trajectory.facecolor = 'b'
    # rnd.draw_params.dynamic_obstacle.trajectory.shape.zorder = 10000
    # rnd.draw_params.dynamic_obstacle.trajectory.shape.zorder = 10000
    print(dps)


    def animate(i):
        rnd.draw_params.time_begin = i
        rnd.draw_params.time_end = i + 15
        scenario.draw(rnd)
        if ego_v:
            ego_dp = copy.deepcopy(rnd.draw_params)
            ego_dp.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor='g'
            ego_v.draw(rnd, draw_params=ego_dp)
        # planning_problem_set.draw(rnd)
        rnd.render()


    ani = FuncAnimation(fig, animate, frames=timesteps, interval=32, repeat=True, repeat_delay=200)
    if save_path is not None:
        ani.save(save_path, animation.PillowWriter(fps=30))
    plt.show()
