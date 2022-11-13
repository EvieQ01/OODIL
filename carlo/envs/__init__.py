# from envs.gridworld_continuous import GridworldContinuousEnv, GridworldContinuousSlowRandomInitEnv, GridworldContinuousFastRandomInitEnv
from gym.envs.registration import register
from assets import *
register(id="Continuous-v0", entry_point="envs.gridworld_continuous:GridworldContinuousEnv",max_episode_steps=1000) # Maxspeed 3.0
register(id="Continuous-v05", entry_point="envs.gridworld_continuous:GridworldContinuousEnvSpeed05",max_episode_steps=1000) # Maxspeed 0.5
register(id="Continuous-v0505025", entry_point="envs.gridworld_continuous:GridworldContinuousEnvSpeed05_1",max_episode_steps=1000) # Maxspeed 0.5
register(id="Continuous-v1", entry_point="envs.gridworld_continuous:GridworldContinuousEnvSpeed1",max_episode_steps=1000) # Maxspeed 1.0
register(id="Continuous-v105025", entry_point="envs.gridworld_continuous:GridworldContinuousEnvSpeed1_1",max_episode_steps=1000) # Maxspeed 1.0
register(id="Continuous-v104025", entry_point="envs.gridworld_continuous:GridworldContinuousEnvSpeed1_2",max_episode_steps=1000) # Maxspeed 1.0
register(id="Continuous-v2", entry_point="envs.gridworld_continuous:GridworldContinuousEnvSpeed2",max_episode_steps=1000) # Maxspeed 2.0
register(id="Continuous-v5", entry_point="envs.gridworld_continuous:GridworldContinuousEnvSpeed5",max_episode_steps=1000) # Maxspeed 5.0
register(id="Continuous-v10", entry_point="envs.gridworld_continuous:GridworldContinuousEnvSpeed10",max_episode_steps=1000) # Maxspeed 5.0
# let the biggest obstacle be the target
register(id="Continuous-v305025", entry_point="envs.gridworld_continuous:GridworldContinuousEnvSpeed3_0",max_episode_steps=1000) # Maxspeed 3.0
register(id="Continuous-v30104", entry_point="envs.gridworld_continuous:GridworldContinuousEnvSpeed3_1",max_episode_steps=1000) # Maxspeed 3.0
register(id="Continuous-v301025", entry_point="envs.gridworld_continuous:GridworldContinuousEnvSpeed3_2",max_episode_steps=1000) # Maxspeed 3.0
