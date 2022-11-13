import time
import pdb
from tkinter.tix import Tree
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from pdenv.gym_panda.panda_bullet.panda import *
from pdenv.gym_panda.panda_bullet.objects import YCBObject, InteractiveObj, RBOObject
import os
import numpy as np
import pybullet as p
import pybullet_data
import copy
import pickle
class PandaEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self, panda_type=Panda, goal_posi=0.91):
    # create simulation (GUI)
    self.urdfRootPath = pybullet_data.getDataPath()
    
    p.connect(p.DIRECT)
    #p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    self._set_camera()

    # load some scene objects
    p.loadURDF(os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
    p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0., -0.65])

    # example YCB object
    self.box_base_position = [ 0.5, -0.,  0.08]
    self.obj_id = None
    self.set_obj()
    self.panda = panda_type()
    self.arm_id = self.panda.panda
    self.n = 3
    self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(self.n, ), dtype=np.float32)
    self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(6,), dtype=np.float32)
    self.goal_posi = np.array([goal_posi, 0, 0.05])
    self.action_mode = 1 # mode = 1 as xyz, mode = 0 as joint position
    # pdb.set_trace()
    # print('using goal', goal_posi)
  
  def set_obj(self, mass=10.):
    if self.obj_id:
      p.removeBody(self.obj_id)
    self.obj1 = YCBObject('push_box', mass=mass)
    self.obj1.load()
    self.obj_id = self.obj1.body_id
    p.resetBasePositionAndOrientation(self.obj_id, self.box_base_position, [0, 0, 0, 1])
    
  def reset(self):
    while(True):
      self.panda.reset()
      if self.panda.state['ee_position'][0] < self.box_base_position[0] - 0.1:
        break
    # self.panda.reset()  
    self.episode_step = 0
    self.reward = 0
    self.reachgoal = False
    self.maxlength=4000
    self.length=0
    self.init_position = self.panda.state['ee_position']
    p.resetBasePositionAndOrientation(self.obj_id, self.box_base_position, [0, 0, 0, 1])
    return np.concatenate((self.panda.state['ee_position'], np.array(p.getBasePositionAndOrientation(self.obj_id)[0])), axis=None)

  def reset_set_init(self, set_init_joint_rand):
    # self.reset()
    self.panda.reset(set_init_joint_rand)
    return np.concatenate((self.panda.state['ee_position'], np.array(p.getBasePositionAndOrientation(self.obj_id)[0])), axis=None)
  
  def set_initial_state(self, state):
    self.reset()
    self.panda._reset_robot_eeposition(ee_position=state[0][:3])
    return np.concatenate((self.panda.state['ee_position'], np.array(p.getBasePositionAndOrientation(self.obj_id)[0])), axis=None)

  def set_action_mode(self, mode=1):
    # mode = 0 as joint position
    # mode = 1 as xyz
    self.action_mode = mode
    if self.action_mode == 1:
      self.n = 3
    else:
      self.n = len(self.panda.resetpos)
  
  def set_action_space(self):
    self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(self.n, ), dtype=np.float32)

  def resetobj(self):
    print( self.panda.state['ee_position'])
    p.resetBasePositionAndOrientation(self.obj_id, self.panda.state['ee_position']+np.array([0,0.002,0.05]), [0, 0, 0, 1])
    return self.panda.state
  def get_obj_ee_posi(self):
    return p.getBasePositionAndOrientation(self.obj_id)[0]
  def close(self):
    p.disconnect()
  
  def seed(self, seed=None):
        self.panda.seed(seed)
        return [seed]

  def step(self, action, verbose=False, mode=1):
    # get current state
    self.episode_step +=1
    state = self.panda.state

    # TODO
    if self.action_mode == 1:
        # use xyz
        self.panda.step(dposition=action)
    else:
        # use joint actions
        self.panda.step(djoint=action, mode=0)

    # take simulation step
    p.stepSimulation()
    # time.sleep(1./240.)

    # return next_state, reward, done, info
    next_state = self.panda.state
    self.reward = 0
    state = next_state['ee_position']
    obj_position = p.getBasePositionAndOrientation(self.obj_id)[0]
    
    # reward for moved distance
    self.reward += (obj_position[0] - 0.5) * 1 # distance * 10

    self.reachgoal = np.linalg.norm(np.array([obj_position[0] - self.goal_posi[0], 0,obj_position[2] - self.goal_posi[2]])) < 0.05
    done =False
    next_state = np.concatenate((state, np.array(obj_position)), axis=None) # shape 6
    if(self.reachgoal):
      self.reward +=  1000
      print('Done, ', next_state, 'episode steps: ', self.episode_step)
      
      done = True
      self.episode_step = 0
    if ((state[0] - obj_position[0]) > .5) or obj_position[-1] < -0.1 or (np.array(obj_position)).max() > 1.:
      self.reward = -1000
      done = True
      print('miss', next_state, 'episode steps: ', self.episode_step)
      self.episode_step = 0
    info = {}
    info['joint_action'] = self.panda.last_joint_control
    
    return next_state, self.reward, done, info

  def render(self, mode='human', close=False):
    (width, height, pxl, depth, segmentation) = p.getCameraImage(width=self.camera_width,
                                                                    height=self.camera_height,
                                                                    viewMatrix=self.view_matrix,
                                                                    projectionMatrix=self.proj_matrix)
    rgb_array = np.array(pxl, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (self.camera_height, self.camera_width, 4))
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def _set_camera(self):
    self.camera_width = 256
    self.camera_height = 256
    p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-15,
                                    cameraTargetPosition=[0.3, -0.2, 0.0])
    self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.2, 0., -0.],
                                                            distance=1.8,
                                                            yaw=0,
                                                            pitch=-20,
                                                            roll=0,
                                                            upAxisIndex=2)
    self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                    aspect=float(self.camera_width) / self.camera_height,
                                                    nearVal=0.1,
                                                    farVal=100.0)

class PandaEnv1(PandaEnv):

    def __init__(self, panda_type=Panda, goal_posi=0.81):
        super(PandaEnv1, self).__init__(panda_type=panda_type, goal_posi=goal_posi)
        self.panda.init_range = 1.0

class PandaEnv2(PandaEnv):

    def __init__(self, panda_type=Panda, goal_posi=0.81):
        super(PandaEnv2, self).__init__(panda_type=panda_type, goal_posi=goal_posi)
        self.panda.init_range = 2.0

class DisabledPandaEnv(PandaEnv):

    def __init__(self, panda_type=DisabledPanda, goal_posi=0.81):
        super(DisabledPandaEnv, self).__init__(panda_type=panda_type, goal_posi=goal_posi)
        self.action_space = spaces.Box(low=np.array([-1., -1.]), high=np.array([1., 1.]), dtype=np.float32)

    def step(self, action,verbose=False):
        action = action.squeeze()
        action1 = np.array([0., 0., 0.])
        action1[0] = action[0]
        action1[2] = action[2]   #change to 1 for rl, 2 for collect demons
        next_state = self.panda.state

        # set action1[1] to zero 
        return super(DisabledPandaEnv, self).step(action1, verbose)
class DisabledPandaEnv4(PandaEnv):

    def __init__(self, panda_type=DisabledPanda4, goal_posi=0.81):
        super(DisabledPandaEnv4, self).__init__(panda_type=panda_type, goal_posi=goal_posi)
        self.set_action_mode(mode=0)

    def step(self, action,verbose=False):
      if self.action_mode == 0:
        action = action.squeeze()
        action_true = np.zeros(7)
        for i in range(len(self.panda.resetpos)):
          # self.panda.resetpos = [0,1,2,4,5]
          # self.panda.resetpos[i] is the joint that needs to apply action.
          action_true[self.panda.resetpos[i]] = action[i]
        
        return super(DisabledPandaEnv4, self).step(action_true, verbose)
      else:
        return super(DisabledPandaEnv4, self).step(action, verbose)

class DisabledPandaEnv1(DisabledPandaEnv4):

    def __init__(self, panda_type=DisabledPanda1, goal_posi=0.81):
        super(DisabledPandaEnv1, self).__init__(panda_type=panda_type, goal_posi=goal_posi)
        # self.set_action_mode(mode=0)
        # self.action_space = spaces.Box(low=-1. * np.ones(6), high=1. * np.ones(6), dtype=np.float32)

class DisabledPandaEnv6(DisabledPandaEnv4):

    def __init__(self, panda_type=DisabledPanda6, goal_posi=0.81):
        super(DisabledPandaEnv6, self).__init__(panda_type=panda_type, goal_posi=goal_posi)

class DisabledPandaEnv3(DisabledPandaEnv4):

    def __init__(self, panda_type=DisabledPanda3, goal_posi=0.81):
        super(DisabledPandaEnv3, self).__init__(panda_type=panda_type, goal_posi=goal_posi)

class DisabledPandaEnv13(DisabledPandaEnv4):

    def __init__(self, panda_type=DisabledPanda13, goal_posi=0.81):
        super(DisabledPandaEnv13, self).__init__(panda_type=panda_type, goal_posi=goal_posi)

class DisabledPandaEnv14(DisabledPandaEnv4):

    def __init__(self, panda_type=DisabledPanda14, goal_posi=0.81):
        super(DisabledPandaEnv14, self).__init__(panda_type=panda_type, goal_posi=goal_posi)

class DisabledPandaEnv134(DisabledPandaEnv4):

    def __init__(self, panda_type=DisabledPanda134, goal_posi=0.81):
        super(DisabledPandaEnv134, self).__init__(panda_type=panda_type, goal_posi=goal_posi)

class DisabledPandaEnv1346(DisabledPandaEnv4):
    def __init__(self, panda_type=DisabledPanda1346, goal_posi=0.81):
        super(DisabledPandaEnv1346, self).__init__(panda_type=panda_type, goal_posi=goal_posi)


class DisabledPandaEnv1357(DisabledPandaEnv4):
    def __init__(self, panda_type=DisabledPanda1357, goal_posi=0.81):
        super(DisabledPandaEnv1357, self).__init__(panda_type=panda_type, goal_posi=goal_posi)



class FeasibilityPandaEnv(PandaEnv):
  def __init__(self, panda_type=FeasibilityPanda, goal_posi=0.81):
    super(FeasibilityPandaEnv, self).__init__(panda_type=panda_type, goal_posi=goal_posi)
    # self.action_space = spaces.Box(low=np.array([-1., -1.]), high=np.array([1., 1.]), dtype=np.float32)
    self.action_space = spaces.Box(low=np.array([-1., -1., -1.]), high=np.array([1., 1., -1.]), dtype=np.float32)
    self.observation_space = spaces.Box(low=np.array([-1., -1., -1.]), high=np.array([1., 1., -1.]), dtype=np.float32)

    # self.gt_data1 =  pickle.load(open('../data/dis5.pkl', 'rb'))
    # self.gt_data2 =  pickle.load(open('../data/normal48.pkl', 'rb'))
    self.time_step = 0
    self.reward = 0
    self.eps_len = 8000
    self.episode_step = 0
  def set_gt_demos(self, demos):
    self.gt_data = demos

  def _random_select(self, idx=None, version=None):
        if idx is None:
            self.gt_num = np.random.choice(self.gt_data.shape[0])
        else:
            self.gt_num = idx
        self.gt = self.gt_data[self.gt_num][:][:]
        eeposition = self.gt[0][:3]
        self.panda._reset_robot_eeposition(ee_position=eeposition)
        self.eps_len = self.gt.shape[0]
  
    
  def reset(self,idx=None, version = None):
    self.panda.reset()
    self._random_select(idx, version)
    self.time_step = 0
    self.episode_step = 0
    self.reward = 0
    self.reachgoal = False
    self.maxlength=4000
    self.length=0
    self.init_position = self.panda.state['ee_position']
    p.resetBasePositionAndOrientation(self.obj_id, self.box_base_position, [0, 0, 0, 1])
    return self.panda.state['ee_position']
    return state
  

  def step(self, action,verbose=False):
    action = action.squeeze()
    self.episode_step +=1
    state = self.panda.state

    # TODO
    if self.action_mode == 1:
      # use xyz
      self.panda.step(dposition=action)
    else:
      # use joint actions
      self.panda.step(djoint=action, mode=0)

    # take simulation step
    p.stepSimulation()
    state = self.panda.state['ee_position']
    self.time_step += 1
    done = (self.time_step >= self.eps_len - 1)
    dis = np.linalg.norm(state - self.gt[self.time_step][:3])
    reward = -dis
    info = {}
    self.prev_state = copy.deepcopy(state)
    full_state = self.panda.state['ee_position']
    info['dis'] = dis
        
    return full_state, reward, done, info

class FeasibilityPandaEnv13(FeasibilityPandaEnv):
  def __init__(self, panda_type=FeasibilityPanda13, goal_posi=0.81):
    super(FeasibilityPandaEnv13, self).__init__(panda_type=panda_type, goal_posi=goal_posi)
  def reset(self, idx=None, version=None):
      return super().reset(idx, version)

class FeasibilityPandaEnv134(FeasibilityPandaEnv):
  def __init__(self, panda_type=FeasibilityPanda134, goal_posi=0.81):
    super(FeasibilityPandaEnv134, self).__init__(panda_type=panda_type, goal_posi=goal_posi)
  def reset(self, idx=None, version=None):
      return super().reset(idx, version)
