from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from pymapf.decentralized.nmpc.nmpc import MultiAgentNMPC
from pymapf.decentralized.position import Position
# import numpy as np

sim = MultiAgentNMPC()
sim.register_agent("r2d2", Position(1, 3), Position(19, 7), vmin=0)  # pair A
sim.register_agent("bb8", Position(1, 7), Position(5, 19), vmin=0)  # pair B
sim.register_agent("c3po", Position(19, 7), Position(1, 3), vmin=0)  # pair A
sim.register_agent("r4d4", Position(19, 3), Position(1, 13), vmin=0)  # pair C
sim.register_agent("wally", Position(5, 19), Position(1, 7), vmin=0)  # pair B
sim.register_agent("spot", Position(1, 13), Position(19, 3), vmin=0)  # pair C
sim.run_simulation()
sim.visualize("switch_positions_nmpc", 20, 20)
