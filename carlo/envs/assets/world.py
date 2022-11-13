from typing import Union
import numpy as np
from assets.agents import Car, Pedestrian, Building
from assets.entities import Entity
from assets.visualizer import Visualizer


class World:
    def __init__(self, dt: float, width: float, height: float, ppm: float = 8):
        self.dynamic_agents = []
        self.static_agents = []
        self.t = 0  # simulation time
        self.dt = dt  # simulation time step
        self.visualizer = Visualizer(width, height, ppm=ppm)

    def add(self, entity: Entity):
        if entity.movable:
            self.dynamic_agents.append(entity)
        else:
            self.static_agents.append(entity)

    def tick(self):
        for agent in self.dynamic_agents:
            agent.tick(self.dt)
        self.t += self.dt

    def render(self, correct_pos=None, next_pos=None):
        self.visualizer.create_window(bg_color="white")
        self.visualizer.update_agents(self.agents, correct_pos, next_pos)

    @property
    def state(self):
        return np.concatenate([agent.state for agent in self.dynamic_agents])

    @state.setter
    def state(self, x):
        num_agents = len(self.dynamic_agents)
        assert x.shape[0] % num_agents == 0
        agent_state_length = int(x.shape[0] / num_agents)
        offset = 0
        for agent in self.dynamic_agents:
            agent_new_state = x[offset : offset + agent_state_length]
            agent.state = agent_new_state
            offset += agent_state_length

    @property
    def agents(self):
        return self.static_agents + self.dynamic_agents

    def collision_exists(self, agent=None):
        if agent is None:
            for i in range(len(self.dynamic_agents)):
                for j in range(i + 1, len(self.dynamic_agents)):
                    if self.dynamic_agents[i].collidable and self.dynamic_agents[j].collidable:
                        if self.dynamic_agents[i].collidesWith(self.dynamic_agents[j]):
                            return True
                for j in range(len(self.static_agents)):
                    if self.dynamic_agents[i].collidable and self.static_agents[j].collidable:
                        if self.dynamic_agents[i].collidesWith(self.static_agents[j]):
                            return True
            return False

        if not agent.collidable:
            return False

        for i in range(len(self.agents)):
            if (
                self.agents[i] is not agent
                and self.agents[i].collidable
                and agent.collidesWith(self.agents[i])
            ):
                return True
        return False

    def close(self):
        self.reset()
        self.static_agents = []
        self.visualizer.close()

    def reset(self):
        self.dynamic_agents = []
        self.static_agents = []
        self.t = 0
