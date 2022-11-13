# from tkinter.tix import Tree
import argparse
import matplotlib
from matplotlib import pyplot as plt
from sympy import arg
from yaml import parse

matplotlib.use('Agg') 
import pickle
import pdb
from random import random
# import tkinter
import numpy as np
from entities import Entity
from world import World
from agents import Car, RingBuilding, CircleBuilding, Painting, Pedestrian, RectangleBuilding, RectangleEntity
from geometry import Point
import time
# from tkinter import *
import random
from tqdm import trange
import sys
sys.path.append('../')
from setup.models import Policy
import torch
from typing import Tuple
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

def is_in_gap(car:Car, obstacles, eps=0.1):
# def is_in_gap(car:Car, obstacles:list[RectangleEntity], eps=0.1):
    for obstacle in obstacles:
        if obstacle.left - eps < car.right and obstacle.right + eps > car.left:
            return False
    return True
    
def get_nearest_gap(car:Car, obstacles):
    min_dist = 10000.
    for obstacle in obstacles:
        
        if abs(car.center.x - obstacle.left) < abs(min_dist) and obstacle.left > 2.:
            min_dist = abs(car.center.x - obstacle.left)
        if abs(car.center.x - obstacle.right) < abs(min_dist) and obstacle.right < args.world_width - 2:
            min_dist = -abs(car.center.x - obstacle.right)
    
    # return the minimal distance to all obstacles
    # positive, car is relatively on the left
    # negative, car is relatively on the right
    return min_dist #+ car.rear_dist * np.pi

def is_safe(car:Car, obstacles, eps=10):
    for obs in obstacles:
        if -car.y + obs.y < eps and -car.y + obs.y  > 0:
            return False
        # if car.distanceTo(obs) < eps:
        #     return False
    return True

def out_of_map(car:Car, width, height):
    return car.x > width or car.x < 0 or car.y > height or car.y < 0

def visualize_states(demos):
    # counts * steps * states
    # 500 * 128 * 7
    states_xy = np.concatenate(demos)[:, 0:2]
    plt.figure()
    plt.scatter(x=states_xy[:, 0], y=states_xy[:, 1], s=1)
    if args.obstacle_ratio == [0.25, 0.25]:
        plt.savefig(f'../demo/Maxspeed_{args.max_speed}_{args.init_range}.png')
    else:
        plt.savefig(f'../demo/Maxspeed_{args.max_speed}_{args.obstacle_ratio}_{args.init_range}.png')

def make_demo(times, args):
    raw_demos = {}
    raw_demos['obs'] = []
    # raw_demos['obs_normed'].append(expert_traj_normed)
    raw_demos['next_obs'] = []
    raw_demos['action'] = []
    num_steps = 0
    if 'IL' in args.mode:
        policy_net = Policy(6, 1, 64)#.to(device)
        print("=> load: ",args.load_path)
        model_dict = torch.load(args.load_path)
        policy_net.load_state_dict(model_dict['policy'])

    for t in trange(times):
        w = World(args.dt, width = args.world_width, height = args.world_height, ppm = 6,video_path=f'{args.mode}_episode{t}.mp4') # The world is 120 meters by 120 meters. ppm is the pixels per meter.
        # Let's add some sidewalks and RectangleBuildings.
        # A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks / zebra crossings / or creating lanes.
        # A CircleBuilding or RingBuilding object is also static -- they do not move. But as opposed to Painting, they can be collided with.

        # To create a circular road, we will add a CircleBuilding and then a RingBuilding around it
        obstacle1 = RectangleBuilding(center=Point(args.world_width * args.obstacle_ratio[0]/2, args.world_height/2), size=Point(args.world_width * args.obstacle_ratio[0], 2), color='blue')
        w.add(obstacle1)
        obstacle2 = RectangleBuilding(center=Point(3 * args.world_width/4, args.world_height/2), size=Point(args.world_width* args.obstacle_ratio[1], 5), color='blue')
        w.add(obstacle2)
        obstacles = [obstacle1, obstacle2]

        ## create initial states
        init_line = Painting(center=Point(args.world_width/2, args.world_height/10), size=Point(args.world_width  - 2, 1))
        w.add(init_line)
        # create final states
        final_line = RectangleBuilding(center=Point(args.world_width / 2, 9 * args.world_height / 10), size=Point(args.world_width - 2, 1))
        w.add(final_line)

        if args.init_range == 'left':
            init_posi  = Point(random.randint(0, 15), init_line.center.y)
        elif args.init_range == 'right':
            init_posi  = Point(random.randint(45, 60), init_line.center.y)
        elif args.init_range == 'middle':
            init_posi  = Point(random.randint(22.5, 37.5), init_line.center.y)
        elif args.init_range  == 'all':
            cluster_type = random.randint(0,2) # select : left, middle, right
            init_posi  = Point(random.randint(25 * cluster_type, 25 * cluster_type + 10), init_line.center.y)

            # init_posi  = Point(init_line.center.x + random.randint(- init_line.size.x / 2, init_line.size.x / 2), init_line.center.y)
        c1 = Car(center=init_posi, heading=np.pi/2 + (np.random.random() - 0.5))
        c1.max_speed = args.max_speed # let's say the maximum is 30 m/s (108 km/h)
        c1.velocity = Point(0,args.max_speed)
        # c1.velocity = Point(0, min(Max_speed, 3.0))
        acccelarator = PidVelPolicy(dt=args.dt)
        w.add(c1)
        if args.render:
            w.render() # This visualizes the world we just constructed.
        expert_traj = []
        actions = []
        next_state = []
        if args.mode == 'expert':
            # judge which center to choose
            for episode_steps in range(1000):
                # lp = 0.
                last_steering = 0
                if is_in_gap(c1, obstacles, eps=0.1):
                    if c1.heading > 1.5 * np.pi: # right, likely to turn around,should be (-pi/2, 0)
                        steer = np.pi/2 -  c1.heading +  1.9* np.pi
                    elif c1.heading > np.pi: # left, likely to turn around, steer is in ()
                        steer = -1
                    else:
                        steer = np.pi/2. - c1.heading
                    if args.debugging:
                        print('Is in gap. Try to go straight with steering: ', steer,'\theading: ', c1.heading)
                else:
                    if args.debugging:
                        print('Cannot go straight')
                    min_dist = get_nearest_gap(c1, obstacles=obstacles) 
                    # min_dist /= 10
                    if args.debugging:
                        print('min_distance to obstacles:', min_dist)
                    # lp += abs(min_dist) 
                    if is_safe(c1, obstacles, eps=10): 
                    # control 2, with possibliy lp
                    # likely to happen when distance is big
                        if args.debugging:
                            print("safe : )")
                        if random.random() < 1. and c1.heading < np.pi * 0.9 and c1.heading > 0.1 * np.pi:
                            max_obstacle = max(obstacle1.size.x, obstacle2.size.x)
                            steer = min_dist / max_obstacle * (np.pi / 2) + 0.1 if min_dist > 0. else \
                                min_dist / max_obstacle * (np.pi / 2) - 0.1
                            # steer = np.clip(steer, a_min=np.pi - c1.heading, a_max=c1.heading)
                            if args.debugging:
                                print("Steering: ", steer)
                        else: 
                            steer = np.pi/2. - c1.heading
                            # steer = np.clip(steer, a_min=np.pi - c1.heading, a_max=c1.heading)
                            steer = 0 if ((np.pi - c1.heading) < 1e-5 or c1.heading < 1e-5) else steer
                            if args.debugging:
                                print('Try to go straight with steering: ', steer)
                    else:
                        if args.debugging:
                            print("not safe!!!")
                        steer = min_dist / max_obstacle * (np.pi / 2) + 0.1 if min_dist > 0 else \
                                    min_dist / max_obstacle * (np.pi / 2) - 0.1
                        steer = 0 if c1.heading < 1e-5 else steer
                        steer = 0 if (np.pi - c1.heading) < 0.2 else steer
                        if args.debugging:
                            print("Steering: ", steer)
                accel = 10.

                c1.set_control(steer, accel)
                expert_traj.append(c1.state)
                actions.append(steer)
                w.tick() # This ticks the world for one time step (dt second)
                if args.render:
                    w.render()
                    # time.sleep(dt/4) # Let's watch it 4x
                # save #


                if w.collision_exists() or out_of_map(c1, args.world_width, args.world_height): # We can check if there is any collision at all.
                    print('Collision exists somewhere...')
                    done = True
                    # w.close()
                    if c1.collidesWith(final_line):
                        print("steps in this episode: ", episode_steps)
                        print(f"total steps: {num_steps}/{args.total_steps}")
                        # print("save traj[{}] with rewards: {}".format(len(raw_demos['obs']), reward_episode))
                        raw_demos['obs'].append(np.array(expert_traj))
                        # raw_demos['obs_normed'].append(expert_traj_normed)
                        raw_demos['next_obs'].append(np.array(expert_traj[1:] + c1.state))
                        raw_demos['action'].append(np.array(actions))
                        num_steps += episode_steps
                        
                        if len(raw_demos['obs']) == 1:
                            print("No.1 traj:", raw_demos['obs'][0][0])
                        if len(raw_demos['obs']) == args.count_of_demos or num_steps >= args.total_steps:
                            
                            if args.visualize:
                                visualize_states(raw_demos['obs'])
                            return raw_demos
                    break
            # w.close()

        elif args.mode == 'human': # Let's use the keyboard input for human control
            from interactive_controllers import KeyboardController
            c1.set_control(0., 0.) # Initially, the car will have 0 steering and 0 throttle.
            controller = KeyboardController(w)
            for k in range(600):
                c1.set_control(controller.steering, controller.throttle)
                w.tick() # This ticks the world for one time step (dt second)
                w.render()
                time.sleep(args.dt) # Let's watch it 4x
                if w.collision_exists():
                    import sys
                    sys.exit(0)
        else:# loaded policy
            for episode_steps in range(1000):
                accel = 10.
                state = c1.state[:6]
                state[0] = state[0] / 60 * 2 - 1
                state[1] = state[1] / 50 * 2 - 1
                with torch.no_grad():
                    steer = policy_net(torch.tensor(state,dtype=torch.float32), True)
                # print(state)
                c1.set_control(steer.detach().data[0], accel)
               
                w.tick() # This ticks the world for one time step (dt second)
                if args.render and episode_steps % 5 == 0:
                    w.render()
                    time.sleep(args.dt/2) # Let's watch it 4x
                # save #


                if w.collision_exists() or out_of_map(c1, args.world_width, args.world_height): # We can check if there is any collision at all.
                    print('Collision exists somewhere...')
                    w.videoWriter.release() #释放    @property

                    break

        w.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    parser.add_argument('--mode', default='IL', type=str, help='the policy to make demonstrations')
    parser.add_argument('--count', default=1, type=int, help='how many wimes to rollout')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--load_path', default='../setup/log/Continuous-v104025/3GAIL_simclr_dcn_N3_source_1.0_[0.5, 0.25]_1.0_[0.1, 0.5]_5.0_ratio_[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]/checkpoints/episode_1900_seed_1111_gail_model.pth',
                        help='the trained imitation learning policy', type=str)
    parser.add_argument('--count_of_demos', default=100000, type=int,help='max trajectory count')
    parser.add_argument('--total_steps', default=5000000, type=int, help='max step count')
    parser.add_argument('--init_range', default='all', type=str)
    
    
    parser.add_argument('--obstacle_ratio', nargs='+', default=[0.4, 0.25])
    parser.add_argument('--world_height', default=50, type=int)
    parser.add_argument('--world_width', default=60, type=int)
    parser.add_argument('--max_speed', default=1., type=float)
    parser.add_argument('--dt', default=.1, type=float,help='time sleep in terms of second'
)


    parser.add_argument('--debugging', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--dump', action='store_true',help='whether to save demonstrations')
    parser.add_argument('--visualize', action='store_true',help='visualize trajectory as images')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    raw_demos = make_demo(args.count, args)

    if args.dump:
        if args.obstacle_ratio == [0.25, 0.25]:
            save_demo_path = f'../demo/Maxspeed_{args.max_speed}_{args.init_range}_batch00.pkl'
        else:
            save_demo_path = f'../demo/Maxspeed_{args.max_speed}_{args.obstacle_ratio}_{args.init_range}_batch00.pkl'
        pickle.dump(raw_demos,
                    open(save_demo_path, 'wb'))
