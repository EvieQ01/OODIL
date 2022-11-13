import pdb

from matplotlib.image import pil_to_array
from agents import Car, Pedestrian, RectangleBuilding
from entities import Entity
from typing import Union
from visualizer import Visualizer
import numpy as np
import cv2
class World:
    def __init__(self, dt: float, width: float, height: float, ppm: float = 8,video_path=''):
        self.dynamic_agents = []
        self.static_agents = []
        self.t = 0 # simulation time
        self.dt = dt # simulation time step
        self.visualizer = Visualizer(width, height, ppm=ppm)
        size=(3968,2976)

        #可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # avi
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4
        self.videoWriter = cv2.VideoWriter(video_path,fourcc,20,(710,600))  

    def add(self, entity: Entity):
        if entity.movable:
            self.dynamic_agents.append(entity)
        else:
            self.static_agents.append(entity)
        
    def tick(self):
        for agent in self.dynamic_agents:
            agent.tick(self.dt)
        self.t += self.dt
    
    def render(self):
        self.visualizer.create_window(bg_color = 'white')
        self.visualizer.update_agents(self.agents())
        img = self.visualizer.gettimg()
        
        height, width, channels = pil_to_array(img).shape
        # print(pil_to_array(img).shape)
        self.videoWriter.write(pil_to_array(img)[:,:,:3])
    def agents(self):
        return self.static_agents + self.dynamic_agents
        
    def collision_exists(self, agent = None):
        if agent is None:
            for i in range(len(self.dynamic_agents)):
                for j in range(i+1, len(self.dynamic_agents)):
                    if self.dynamic_agents[i].collidable and self.dynamic_agents[j].collidable:
                        if self.dynamic_agents[i].collidesWith(self.dynamic_agents[j]):
                            return True
                for j in range(len(self.static_agents)):
                    if self.dynamic_agents[i].collidable and self.static_agents[j].collidable:
                        if self.dynamic_agents[i].collidesWith(self.static_agents[j]):
                            return True
            return False
            
        if not agent.collidable: return False
        
        for i in range(len(self.agents)):
            if self.agents[i] is not agent and self.agents[i].collidable and agent.collidesWith(self.agents[i]):
                return True
        return False
    
    def close(self):
        self.reset()
        self.static_agents = []
        if self.visualizer.window_created:
            self.visualizer.close()
        
    def reset(self):
        self.dynamic_agents = []
        self.t = 0