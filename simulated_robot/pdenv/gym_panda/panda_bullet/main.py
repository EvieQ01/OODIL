from env_1 import SimpleEnv
from env_2 import DoubleEnv
from env_3 import TeleopEnv
import numpy as np
import time


env = SimpleEnv()
state = env.reset()
start_time = time.time()
curr_time = time.time() - start_time
while curr_time < 4*np.pi:
    curr_time = time.time() - start_time
    action = [0.01*np.cos(curr_time), 0.01*np.sin(curr_time), 0]
    next_state, reward, done, info = env.step(action)
    # img = env.render()
    if done:
        break
env.close()


env = DoubleEnv()
state = env.reset()
start_time = time.time()
curr_time = time.time() - start_time
while curr_time < 4*np.pi:
    curr_time = time.time() - start_time
    action1 = [0.01*np.cos(curr_time), 0.01*np.sin(curr_time), 0]
    action2 = [0, 0.01*np.cos(curr_time), 0.01*np.sin(curr_time)]
    next_state, reward, done, info = env.step(action1+action2)
    # img = env.render()
    if done:
        break
env.close()


env = TeleopEnv()
state = env.reset()
while True:
    next_state, reward, done, info = env.step()
    # img = env.render()
    if done:
        break
env.close()
