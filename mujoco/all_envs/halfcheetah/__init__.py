from gym.envs.registration import register

register(
    id='CustomHalfCheetah-v0',
    entry_point='halfcheetah.half_cheetah_v3:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
