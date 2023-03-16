import copy
from typing import Optional, List, Dict, Tuple

import numpy as np
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


def animate_scenario(scenario: Scenario, planning_problem_set, timesteps: int, ego_v: Optional[DynamicObstacle] = None,
                     save_path=None, show=False, figax=None):
    if figax is None:
        fig, ax = plt.subplots(figsize=(25, 3))
    else:
        fig, ax = figax

    rnd = MPRenderer(ax=ax)
    dps = rnd.draw_params
    is_s = rnd.draw_params["planning_problem"]["initial_state"]["state"]
    rnd.draw_params.dynamic_obstacle.trajectory.draw_continuous = True
    rnd.draw_params.dynamic_obstacle.show_label = True

    def animate(i):
        rnd.draw_params.time_begin = i
        rnd.draw_params.time_end = i + 15
        scenario.draw(rnd)
        if ego_v:
            ego_dp = copy.deepcopy(rnd.draw_params)
            ego_dp.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = 'g'
            ego_v.draw(rnd, draw_params=ego_dp)
        # planning_problem_set.draw(rnd)
        rnd.render()
        # rnd.ax.scatter([5], [1.75], c='b', zorder=100)

    ani = FuncAnimation(fig, animate, frames=timesteps, interval=32, repeat=True, repeat_delay=200)
    if save_path is not None:
        ani.save(save_path, animation.PillowWriter(fps=30))
    if show:
        plt.show()
    return ani


def ani_scenario_func(scenario: Scenario, ax, ego_v: Optional[DynamicObstacle] = None):
    rnd = MPRenderer(ax=ax)
    rnd.draw_params.dynamic_obstacle.trajectory.draw_continuous = True
    rnd.draw_params.dynamic_obstacle.show_label = True

    def animate(i):
        rnd.draw_params.time_begin = i
        rnd.draw_params.time_end = i + 15
        scenario.draw(rnd)
        if ego_v:
            ego_dp = copy.deepcopy(rnd.draw_params)
            ego_dp.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = 'g'
            ego_v.draw(rnd, draw_params=ego_dp)
        rnd.render()

    return animate


def ani_multi_monitor_func(ob_rob_vs, ax):
    ob_lines = {}
    for ob_id, rob_vs in ob_rob_vs.items():
        ob_lines[ob_id] = ax.plot(rob_vs, label=ob_id)[0]

    def animate(i):
        for ob_id, rob_vs in ob_rob_vs.items():
            ob_lines[ob_id].set_data(range(i), rob_vs[:i])
        return list(ob_lines.values())[0]

    ax.legend(loc='best')

    return animate


def ani_monitor_func(rob_vs, ax):
    line = ax.plot(rob_vs)[0]

    def animate(i):
        line.set_data(range(i), rob_vs[:i])
        return line

    return animate


def animate_multi_monitor(ob_rob_vs: Dict[int, List[float]], fig_ax: Tuple = None, show=False):
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=(25, 3))
    else:
        fig, ax = fig_ax

    y_min, y_max = -1, -np.inf
    for ob_id, rob_vs in ob_rob_vs.items():
        time_steps = len(rob_vs)
        y_min = min(y_min, np.min(rob_vs) - 0.5)
        y_max = max(y_max, np.max(rob_vs) + 0.5)

    animate = ani_multi_monitor_func(ob_rob_vs, ax)

    ax.set_xlim(0, time_steps)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='best')
    ax.set_xlabel("t")
    ax.set_ylabel("Robustness")

    ani = FuncAnimation(fig, animate, frames=time_steps, interval=32, repeat=True, repeat_delay=200)
    if show:
        plt.show()
    return ani


def animate_scenario_and_monitor(scenario: Scenario, rob_vs, save=False):
    fig, axs = plt.subplots(figsize=(25, 6), nrows=2, ncols=1)
    timesteps = len(rob_vs)

    a0 = ani_scenario_func(scenario, axs[0])
    a1 = ani_monitor_func(rob_vs, axs[1])

    def animate(i):
        a0(i)
        a1(i)

    axs[1].set_ylim(min(-1, min(rob_vs) - 1), max(rob_vs) + 1)
    axs[1].plot(range(len(rob_vs)), np.zeros(len(rob_vs)), linestyle='--', c='r', alpha=0.2)

    ani = FuncAnimation(fig, animate, frames=timesteps, interval=32, repeat=True, repeat_delay=300)
    if save:
        ani.save("results/rg_1_monitor.gif", animation.PillowWriter(fps=15))
    plt.tight_layout()
    plt.show()
