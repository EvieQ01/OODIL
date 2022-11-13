import io
import pdb
from typing import Text
import gym
from gym import spaces
from PIL import Image
import numpy as np
import scipy.special
from assets.world import World
from assets.entities import TextEntity, Entity
from assets.agents import Car, Building, Goal, Painting
from assets.geometry import Point
from typing import Tuple

import random

class PidVelPolicy:
    """PID controller for H that maintains its initial velocity."""

    def __init__(self, dt: float, params: Tuple[float, float, float] = (3.0, 1.0, 6.0)):
        self._target_vel = None
        self.previous_error = 0
        self.integral = 0
        self.errors = []
        self.dt = dt
        self.Kp, self.Ki, self.Kd = params

    def action(self, obs):
        my_y_dot = obs[3]
        if self._target_vel is None:
            self._target_vel = my_y_dot
        error = self._target_vel - my_y_dot
        derivative = (error - self.previous_error) * self.dt
        self.integral = self.integral + self.dt * error
        acc = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        self.errors.append(error)
        return acc

    def reset(self):
        self._target_vel = None
        self.previous_error = 0
        self.integral = 0
        self.errors = []

    def __str__(self):
        return "PidVelPolicy({})".format(self.dt)

class GridworldContinuousEnv(gym.Env):

    def __init__(self,
                 dt: float = 0.1,
                 width: int = 60,
                 height: int = 50,
                 time_limit: float = 1000.0):
        super(GridworldContinuousEnv, self).__init__()
        self.dt = dt
        self.width = width
        self.height = height
        self.world = World(self.dt, width=width, height=height, ppm=6)
        self.accelerate = PidVelPolicy(self.dt)
        self.step_num = 0
        self.time_limit = time_limit
        self.action_space = spaces.Box(
            np.array([-1.]), np.array([1.]), dtype=np.float32
        )
        # self.goal_radius = 2.
        self.obs_dim = 6 # x, y, dx, dy, heading, acc
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,))
        self.goal_posi = Point(self.width / 2, 8.5 * self.height / 10)
        # self.start = np.array([self.width/2.,self.goal_radius])
        # self.goal = np.array([self.width/2., self.height-self.goal_radius])
        # self.max_dist = np.linalg.norm(self.goal-self.start,2)

        # self.target = [self.height/5., self.height*2./5., self.height*3./5., self.height*4./5., np.inf]
        self.obstacle_width = [0.25, 0.25]
        self.obstacle_center = [0.25, 0.75]
        self.initial_speed = 3.
        self.max_speed = 3.

    def step(self, action: np.ndarray, verbose: bool = False):
        self.step_num += 1
        # action is steering
        # accelarate is autoPID
        
        # action = action * 0.1
        car = self.world.dynamic_agents[0]
        acc = 10.
        # acc = self.accelerate.action(self._get_obs())
        # acc = self.accelerate.action(np.array(self.world.state))
        action = np.append(action, acc)
        # if self.stop:
        #     action = np.array([0, -5])
        car.set_control(*action) # only use first two dimensions
        self.world.tick()

        reward = self.reward(verbose)

        done = False
        # collide
        for building in self.buildings:
            if self.car.collidesWith(building):
                done = True
                reward -= 1000
                break
        # out of window
        if car.y >= self.height or car.y <= 0 or car.x <= 0 or car.x >= self.width:
            reward -= 1000
            done = True
        if self.step_num >= self.time_limit:
            done = True
        if self.car.collidesWith(self.goal_obj):
            done = True
            self.stop = True
            
        #if self.step_num < 6:
        #    done = False
        return self._get_obs(), reward, done, {'episode': {'r': reward, 'l': self.step_num}}

    def reset(self, init_range=None):
        self.world.reset()
        self.stop = False
        self.target_count = 0

        if init_range is None:
            init_range = (0, self.width)
        self.buildings = [
            Building(center=Point(self.width * self.obstacle_center[0], self.height / 2), size=Point(self.obstacle_width[0] * self.width, 2), color='blue'),
            Building(center=Point(self.width * self.obstacle_center[1], self.height / 2), size=Point(self.obstacle_width[1] * self.width, 5), color='blue'),
            Painting(center=Point(self.width / 2,  self.height / 10), size=Point(self.width - 2, 1))
        ]
        ## create initial states
        # random_angle = random.random()*2*np.pi TODO
        
        
        cluster_type = random.randint(0,2) # select : left, middle, right
        init_posi  = Point(random.randint(25 * cluster_type, 25 * cluster_type + 10), self.buildings[-1].center.y)

        self.car = Car(init_posi, np.pi/2.+ (np.random.random() - 0.5), "r")
        self.car.velocity = Point(0, self.initial_speed)
        self.car.max_speed = self.max_speed
        # a goal of one line
        self.goal_obj = Goal(center=self.goal_posi , size=Point(self.width  - 2, 1), heading=0.0)

        for building in self.buildings:
            self.world.add(building)
        self.world.add(self.car)
        self.world.add(self.goal_obj)
        
        self.last_heading = np.pi / 2

        self.step_num = 0
        return self._get_obs()

    def reset_with_obs(self, obs, after_norm=True):
        '''
        obs in (-1, 1)
        '''
        self.world.reset()

        self.stop = False
        self.target_count = 0

        self.buildings = [
            Building(center=Point(self.width * self.obstacle_center[0], self.height / 2), size=Point(self.obstacle_width[0] * self.width, 2), color='blue'),
            Building(center=Point(self.width * self.obstacle_center[1], self.height / 2), size=Point(self.obstacle_width[1] * self.width, 5), color='blue'),
            Painting(center=Point(self.width / 2,  self.height / 10), size=Point(self.width - 2, 1))
        ]
        if after_norm:
            init_x = (obs[0]/2.+0.5)*self.width
            init_y = (obs[1]/2.+0.5)*self.height
        else:
            init_x = obs[0]
            init_y = obs[1]
        ## create initial states
        # self.init_line = 
        # random_dis = random.random()*2.
        # random_angle = random.random()*2*np.pi TODO
        init_posi  = Point(init_x, init_y)
        # init_x = self.width / 2
        # init_y = self.height / 10
        self.car = Car(init_posi, obs[4])
        # self.car = Car(init_posi, np.pi/2.+ (np.random.random() - 0.5), "r")
        self.car.velocity = Point(0, self.initial_speed)
        self.car.max_speed = self.max_speed
        # a goal of one line
        self.goal_obj = Goal(center=self.goal_posi, size=Point(self.width  - 2, 1), heading=0.0)

        for building in self.buildings:
            self.world.add(building)
        self.world.add(self.car)
        self.world.add(self.goal_obj)
        
        self.last_heading = np.pi / 2

        self.step_num = 0
        return self._get_obs()

    def _get_obs(self):
        """
        Get state of car
        """
        return_state = np.array(self.world.state)
        #print(return_state)
        return_state[1] = 2.* ((return_state[1] / self.height) - 0.5)
        return_state[0] = 2.* ((return_state[0] / self.width) - 0.5)
        return_state[2] /= self.initial_speed
        return_state[3] /= self.initial_speed
        return return_state[:self.obs_dim]

    def inverse_dynamic(self, state, next_state):
        return (next_state[-2] / np.linalg.norm(self.initial_speed * state[2:4], ord=2))/self.dt

    def reward(self, verbose, weight=10.0):
        dist_rew = -1. # * (self.car.center.distanceTo(self.goal_obj)/self.max_dist)
        coll_rew = 0
        for building in self.buildings:
            if self.car.collidesWith(building):
                coll_rew = -100.
                break

        goal_rew = 0.0
        if self.car.collidesWith(self.goal_obj) and (not self.stop):
            goal_rew = 500.

        extra_rew = 0.

        reward = sum([dist_rew, coll_rew, extra_rew, goal_rew])
        if verbose: print("dist reward: ", dist_rew,
                          "goal reward: ", goal_rew,
                          "coll_rew reward: ", coll_rew,
                          "reward: ", reward)
        return reward

    def render(self,mode):
        self.world.render()

class GridworldContinuousEnvSpeed2(GridworldContinuousEnv):
    def __init__(self,
                 dt: float = 0.1,
                 width: int = 60,
                 height: int = 50,
                 time_limit: float = 1000.0):
        super(GridworldContinuousEnvSpeed2, self).__init__(dt, width, height, time_limit)
        self.initial_speed = 2.
        self.max_speed = 2.

class GridworldContinuousEnvSpeed5(GridworldContinuousEnv):
    def __init__(self,
                 dt: float = 0.1,
                 width: int = 60,
                 height: int = 50,
                 time_limit: float = 1000.0):
        super(GridworldContinuousEnvSpeed5, self).__init__(dt, width, height, time_limit)
        self.initial_speed = 5.
        self.max_speed = 5.

class GridworldContinuousEnvSpeed10(GridworldContinuousEnv):
    def __init__(self,
                 dt: float = 0.1,
                 width: int = 60,
                 height: int = 50,
                 time_limit: float = 1000.0):
        super(GridworldContinuousEnvSpeed10, self).__init__(dt, width, height, time_limit)
        self.initial_speed = 10.
        self.max_speed = 10.
class GridworldContinuousEnvSpeed05(GridworldContinuousEnv):
    def __init__(self,
                 dt: float = 0.1,
                 width: int = 60,
                 height: int = 50,
                 time_limit: float = 1000.0):
        super(GridworldContinuousEnvSpeed05, self).__init__(dt, width, height, time_limit)
        self.initial_speed = 0.5
        self.max_speed = .5

class GridworldContinuousEnvSpeed05_1(GridworldContinuousEnv):
    def __init__(self,
                 dt: float = 0.1,
                 width: int = 60,
                 height: int = 50,
                 time_limit: float = 1000.0):
        super(GridworldContinuousEnvSpeed05_1, self).__init__(dt, width, height, time_limit)
        self.initial_speed = 0.5
        self.max_speed = .5
        self.obstacle_width = [0.5, 0.25]

class GridworldContinuousEnvSpeed1(GridworldContinuousEnv):
    def __init__(self,
                 dt: float = 0.1,
                 width: int = 60,
                 height: int = 50,
                 time_limit: float = 1000.0):
        super(GridworldContinuousEnvSpeed1, self).__init__(dt, width, height, time_limit)
        self.initial_speed = 1.
        self.max_speed = 1.

class GridworldContinuousEnvSpeed1_1(GridworldContinuousEnv):
    def __init__(self,
                 dt: float = 0.1,
                 width: int = 60,
                 height: int = 50,
                 time_limit: float = 1000.0):
        super(GridworldContinuousEnvSpeed1_1, self).__init__(dt, width, height, time_limit)
        self.initial_speed = 1.
        self.max_speed = 1.
        self.obstacle_width = [0.5, 0.25]

class GridworldContinuousEnvSpeed1_2(GridworldContinuousEnv):
    def __init__(self,
                 dt: float = 0.1,
                 width: int = 60,
                 height: int = 50,
                 time_limit: float = 1000.0):
        super(GridworldContinuousEnvSpeed1_2, self).__init__(dt, width, height, time_limit)
        self.initial_speed = 1.
        self.max_speed = 1.
        self.obstacle_width = [0.4, 0.25]
        self.obstacle_center = [0.2, 0.75]



# 0.5_0.25
class GridworldContinuousEnvSpeed3_0(GridworldContinuousEnv):
    def __init__(self,
                 dt: float = 0.1,
                 width: int = 60,
                 height: int = 50,
                 time_limit: float = 1000.0):
        super(GridworldContinuousEnvSpeed3_0, self).__init__(dt, width, height, time_limit)
        self.obstacle_width = [0.5, 0.25]
        self.initial_speed = 3.
        self.max_speed = 3.

# 0.1_0.4
class GridworldContinuousEnvSpeed3_1(GridworldContinuousEnv):
    def __init__(self,
                 dt: float = 0.1,
                 width: int = 60,
                 height: int = 50,
                 time_limit: float = 1000.0):
        super(GridworldContinuousEnvSpeed3_1, self).__init__(dt, width, height, time_limit)
        self.obstacle_width = [0.1, 0.4]
        self.initial_speed = 3.
        self.max_speed = 3.
# 0.1_0.25
class GridworldContinuousEnvSpeed3_2(GridworldContinuousEnv):
    def __init__(self,
                 dt: float = 0.1,
                 width: int = 60,
                 height: int = 50,
                 time_limit: float = 1000.0):
        super(GridworldContinuousEnvSpeed3_2, self).__init__(dt, width, height, time_limit)
        self.obstacle_width = [0.1, 0.25]
        self.initial_speed = 3.
        self.max_speed = 3.
