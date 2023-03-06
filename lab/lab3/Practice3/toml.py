from pathlib import Path
from matplotlib import pyplot as plt
import tomli  # the lib to read config file
import sys
from agent import ProblemSolvingAgent
import plotting
from utils.map import mat2obs, read_map

file_folder = Path(__file__).parent
test_folder = file_folder/'test_cases'
with open(test_folder/'case3.toml', 'rb') as f:
    config = tomli.load(f)
world_config = config['world']

map = read_map(world_config, test_folder)
obstacles = mat2obs(map)
# coordinates of origin and destination
start_point = tuple(world_config['start_point'])
goal_point = tuple(world_config['goal_point'])
agent = ProblemSolvingAgent()
# a = list(agent.neighbours_of(obstacles=obstacles, node=start_point))
path, visited = agent.solve_by_searching(
    obstacles, start_point, goal_point, 'DFS')
print(path)
