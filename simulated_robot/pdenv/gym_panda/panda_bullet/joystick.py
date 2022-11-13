import numpy as np
import pygame

# this was originally written for the Logitech F310 gamepad
# start button -> reset_state
# A button -> close gripper
# B buttom -> open gripper
# Joysticks -> move around robot's end-effector
# LB -> switch to controlling end-effector orientation
# RB -> switch to controlling end-effector position

class Joystick():

    def __init__(self, scale=0.01):
        self._reset_internal_state()
        self._deadband = 0.1
        self._scale_trans = scale
        self._scale_rot = scale*5

    def _reset_internal_state(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.mode = 1
        self.grasp = 1

    def _get_inputs(self):
        self.reset_state = self.gamepad.get_button(7)   # start
        self.gripper_open = self.gamepad.get_button(1)  # B
        self.gripper_close = self.gamepad.get_button(0) # A
        self.mode_trans = self.gamepad.get_button(5)    # LB
        self.mode_rot = self.gamepad.get_button(4)      # RB
        z1 = self.gamepad.get_axis(1)
        z2 = self.gamepad.get_axis(0)
        z3 = -self.gamepad.get_axis(4)
        self.z = np.asarray([z1, z2, z3])
        for idx in range(3):
            if abs(self.z[idx]) < self._deadband:
                self.z[idx] = 0.0

    def _get_mode(self):
        if self.mode and self.mode_rot:
            self.mode = 0
        elif not self.mode and self.mode_trans:
            self.mode = 1

    def get_controller_state(self):
        pygame.event.get()
        self._get_inputs()
        self._get_mode()
        dpos = np.array([0.0, 0.0, 0.0])
        dquat = np.array([0.0, 0.0, 0.0, 0.0])
        if self.mode:
            dpos = self.z*self._scale_trans
        else:
            dquat[0:3] = self.z*self._scale_rot
        if self.gripper_open:
            self.grasp = 1
        elif self.gripper_close:
            self.grasp = 0
        return dict(
            dpos=dpos,
            dquat=dquat,
            grasp=self.grasp,
            reset=self.reset_state,
        )
