from .ppo import PPO
from .sac import SAC, SACExpert
from .gail import GAIL
from .gailfo import GAILFO
from .airl import AIRL
from .airlfo import AIRLFO

ALGOS = {
    'gail': GAIL,
    'gailfo': GAILFO,
    'airl': AIRL,
    'airlfo': AIRLFO
}
