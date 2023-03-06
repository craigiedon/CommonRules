from commonroad.scenario.scenario import Scenario
from commonroad.visualization.mp_renderer import MPRenderer
from matplotlib import pyplot as plt, animation
from matplotlib.animation import FuncAnimation


def animate_scenario(scenario: Scenario, planning_problem_set, timesteps: int, save_path=None):
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
