import os
import pdb
import numpy as np
import matplotlib.pyplot as plt

def visualize_obs(obs, path='demo.png'):
    if isinstance(obs, list):
        obs = np.concatenate(obs)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure()
    plt.scatter(x=obs[:, 0], y=obs[:, 1], s=1)
    plt.xlim(0.2, .7)  # 设置x轴的数值显示范围
    plt.ylim(-0.3, 0.3)  # 设置y轴的数值显示范围
    plt.savefig(path)
