import json

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from matplotlib import pyplot as plt

from TaskConfig import TaskConfig
from immFilter import lat_model, update_sim_noiseless


def run():
    file_path = "scenarios/Complex.xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

    with open("config/task_config.json") as f:
        task_config = TaskConfig(**json.load(f))

    all_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [3, 6, 10]]
    ego_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [6, 10]]
    lane_widths = np.abs((ego_lane_centres[0] - ego_lane_centres[1]) / 2.0)

    lat_models = [
        lat_model(task_config.dt, kd, 7, p_ref, 0.1, 0.1)
        for kd in np.linspace(3.0, 5.0, 3)
        for p_ref in all_lane_centres]

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = list(prop_cycle.by_key()['color'])
    fig, ax = plt.subplots()
    x_start = np.array([0, 0])
    T = 40
    for i, lm in enumerate(lat_models):
        xs = [x_start]
        for t in range(0, T):
            xs.append(update_sim_noiseless(xs[-1], lm))
        xs = np.array(xs)
        ax.plot(xs[:, 0], color=colors[i % 3], alpha=0.75)
        for l_cent in all_lane_centres:
            ax.plot(np.arange(T), np.repeat(l_cent, T), linestyle='--', color='black', alpha=0.1)

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Latitude (m)")
        ax.set_ylim(all_lane_centres[0] - lane_widths, all_lane_centres[-1] + lane_widths)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_bounds([0, T])
        ax.spines['left'].set_bounds([ax.get_ylim()[0] + 0.5, ax.get_ylim()[1] - 0.5])

        # TODO: Stick in the lane markings, x/y lims, starting position as a scatter? etc.,
        # TODO: Stick in X/Y labels
        # TODO: Stick in the alpha cycler so that we can distinguish between the three models
        # TODO: Stick in the tufte tricks to neaten it up
    plt.show()


if __name__ == "__main__":
    run()
