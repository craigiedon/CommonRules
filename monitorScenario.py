# Load the scenario
from commonroad.common.file_reader import CommonRoadFileReader
from utils import animate_scenario

file_path = "scenarios/Complex_Solution.xml"
scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

# Create the rule by:
    # Saving the previous receding horizon mpc as a solution file with the ego saved in...
    # Finding the ego vehicle
    # Finding all the vehicles which are not the ego
    # Creating RG1 as an and with all the obstacles
    # Creating the state dictionary listing (using some clever scenario API manipulation?)

# Monitor the rule by having a loop and increasing the final timestep of signal at each one
# Log these robustness values, and chart them
# Sync the car animation with the robustness value animation?
# Save gif and present at group meeting Thursday