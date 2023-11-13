import json

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import InitialState

from TaskConfig import TaskConfig
from monitorScenario import gen_interstate_rules, InterstateRulesConfig


def run():
    file_path = "scenarios/Complex.xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

    with open("config/task_config.json") as f:
        task_config = TaskConfig(**json.load(f))

    with open("config/interstate_rule_config.json", 'r') as f:
        irc = InterstateRulesConfig(**json.load(f))

    all_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [3, 6, 10]]
    ego_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [6, 10]]
    lane_widths = np.abs((ego_lane_centres[0] - ego_lane_centres[1]) / 2.0)

    start_state = InitialState(position=np.array([15.0 + task_config.car_length / 2.0, ego_lane_centres[0]]),
                               velocity=task_config.v_max - 15,
                               orientation=0, acceleration=0.0, time_step=0)

    ego_car = DynamicObstacle(100, ObstacleType.CAR,
                              Rectangle(width=task_config.car_width, length=task_config.car_length), start_state, None)

    rules = gen_interstate_rules(ego_car, scenario.dynamic_obstacles, all_lane_centres, lane_widths, ego_lane_centres,
                             all_lane_centres[0:1], irc)

    # TODO: Find the folder with all the experiment solutions in it, and get working!
    # TODO: Load Rules
    # TODO: Load Scenario
    # TODO: Convert to state dictionary
    # TODO: Calculate failure rate per rule

    # TODO: Load up a random scenario and see if you can replay it in animation
    # TODO: (Maybe not for ground truth though...) See if you can calculate log likelihood of a given trajectory using the PEMs

if __name__ == "__main__":
    run()