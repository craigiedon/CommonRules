import os
import sys

from commonroad.common.file_reader import CommonRoadFileReader

scenario_file = "scenarios/ZAM_Tutorial_Urban-3_2.xml"

scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()
planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]


