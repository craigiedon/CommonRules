from commonroad.common.file_reader import CommonRoadFileReader

from utils import animate_scenario

file_path = "scenarios/CutIn.xml"
scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

animate_scenario(scenario, planning_problem_set, 80)
