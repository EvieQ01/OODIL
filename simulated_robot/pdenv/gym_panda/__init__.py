import logging
from gym.envs.registration import register
import gym
logger = logging.getLogger(__name__)
def register(id, entry_point, force=True, max_episode_steps=2000):
    env_specs = gym.envs.registry.env_specs
    if id in env_specs.keys():
        if not force:
            return
        del env_specs[id]
    gym.register(
        id=id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps
    )
######### different init range #####################
register(
    id='panda-v0',
    entry_point='pdenv.gym_panda.envs:PandaEnv',
    max_episode_steps=2000
)
register(
    id='panda-v1',
    entry_point='pdenv.gym_panda.envs:PandaEnv1',
    max_episode_steps=2000
)
register(
    id='panda-v2',
    entry_point='pdenv.gym_panda.envs:PandaEnv2',
    max_episode_steps=2000
)



###################### differrent disable joint #######
register(
    id='disabledpanda-v0',
    entry_point='pdenv.gym_panda.envs:DisabledPandaEnv',
    max_episode_steps=2000
)
register(
    id='disabledpanda-v1',
    entry_point='pdenv.gym_panda.envs:DisabledPandaEnv1',
    max_episode_steps=2000
)
register(
    id='disabledpanda-v3',
    entry_point='pdenv.gym_panda.envs:DisabledPandaEnv3',
    max_episode_steps=2000
)
register(
    id='disabledpanda-v4',
    entry_point='pdenv.gym_panda.envs:DisabledPandaEnv4',
    max_episode_steps=2000
)
register(
    id='disabledpanda-v6',
    entry_point='pdenv.gym_panda.envs:DisabledPandaEnv6',
    max_episode_steps=2000
)
register(
    id='disabledpanda-v13',
    entry_point='pdenv.gym_panda.envs:DisabledPandaEnv13',
    max_episode_steps=2000
)
register(
    id='disabledpanda-v14',
    entry_point='pdenv.gym_panda.envs:DisabledPandaEnv14',
    max_episode_steps=2000
)
register(
    id='disabledpanda-v134',
    entry_point='pdenv.gym_panda.envs:DisabledPandaEnv134',
    max_episode_steps=2000
)
register(
    id='disabledpanda-v1346',
    entry_point='pdenv.gym_panda.envs:DisabledPandaEnv1346',
    max_episode_steps=2000
)

register(
    id='realpanda-v0',
    entry_point='pdenv.gym_panda.envs:RealPandaEnv',
    max_episode_steps=4000
)