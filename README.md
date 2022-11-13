# Out-of-Dynamics Imitation Learning from Multimodal Demonstrations

##  1. MuJoCo environments.
The implementation for MuJoCo environments is in [mujoco/](mujoco/README.md).

**Acknowledgement**
* [[1]](#reference)This repo is based on [Learning-Feasibility-Different-Dynamics](https://github.com/Stanford-ILIAD/Learning-Feasibility-Different-Dynamics).

* Contrastive clustering algorithm is based on [Deep Clustering Network](https://github.com/xuyxu/Deep-Clustering-Network).
##  2. Driving environment.
The implementation for Driving environment is in [carlo/](carlo/README.md).

**Acknowledgement**
* This repo is based on https://github.com/Stanford-ILIAD/CARLO

## 3. Simulated Franka Panda Arm.

The implementation for Simulated Franka Panda Arm is in [simulated_robot/](simulated_robot/README.md).

**Acknowledgement**
* This repo is based on https://github.com/ku2482/gail-airl-ppo.pytorch


**You can resort to wandb to login your personal account via export your own wandb api key**.
```
export WANDB_API_KEY=YOUR_WANDB_API_KEY
```
and run
```
wandb online
```
to turn on the online syncronization.

### References
[[1]](https://arxiv.org/abs/2110.15142) Z. Cao, Y. Hao, M. Li, and D. Sadigh. Learning feasibility to imitate demonstrators with different dynamics. In CoRL, 2021.


[[2]](https://arxiv.org/abs/1707.06347) Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).

[[3]](http://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning) Ho, Jonathan, and Stefano Ermon. "Generative adversarial imitation learning." Advances in neural information processing systems. 2016.

[[4]](https://arxiv.org/abs/1710.11248) Fu, Justin, Katie Luo, and Sergey Levine. "Learning robust rewards with adversarial inverse reinforcement learning." arXiv preprint arXiv:1710.11248 (2017).

