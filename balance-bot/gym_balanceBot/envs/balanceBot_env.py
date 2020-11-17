import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random

class BalanceBotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.observation = []
        self.action_space = spaces.Discrete(9)

        # pitch, gyro, commanded speed (3 dimensions)
        self.observation_space = spaces.Box(low=np.array([-math.pi, -math.pi, -5]),
                                            high=np.array([math.pi, math.pi, 5])
                                            ) 

        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
        self.seed()

    def step(self, action):
        self.assign_throttle(action)
        p.stepSimulation()
        self.observation = self.computeobservation()
        reward = self.compute_reward()
        done = self.compute_done()
 
        self.envStepCounter += 1
    
        return np.array(self.observation), reward, done, {}

    def reset(self):
        # Current speed
        self.vt = 0
        # Desired speed
        self.vd = 0

        self.envStepCounter = 0

        p.resetSimulation()
        p.setGravity(0,0,-10) # m/s^2
        p.setTimeStep(0.01) # sec
    
        planeId = p.loadURDF("plane.urdf")
        cubeStartPos = [0,0,0.001]
        cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
    
        path = os.path.abspath(os.path.dirname(__file__))
        self.botId = p.loadURDF(os.path.join(path, "../models/balanceBot_simple.urdf"),
                        cubeStartPos,
                        cubeStartOrientation)
    
        self.observation = self.computeobservation()
        return np.array(self.observation)

    def render(self, mode='human', close=False):
        return

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        p.disconnect()

    def assign_throttle(self, action):
        dv = 0.1
        deltav = [-10.*dv,-5.*dv, -2.*dv, -0.1*dv, 0, 0.1*dv, 2.*dv,5.*dv, 10.*dv][action]
        vt = self.vt + deltav
        self.vt = vt
        p.setJointMotorControl2(bodyUniqueId=self.botId, 
                                jointIndex=0, 
                                controlMode=p.VELOCITY_CONTROL, 
                                targetVelocity=vt)
        p.setJointMotorControl2(bodyUniqueId=self.botId, 
                                jointIndex=1, 
                                controlMode=p.VELOCITY_CONTROL, 
                                targetVelocity=-vt)

    def computeobservation(self):
        # Return correspond to observation space
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.botId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        linear, angular = p.getBaseVelocity(self.botId)
        return [cubeEuler[0],angular[0],self.vt]

    def compute_reward(self):
        # cubeEuler 0 -> Upright position
        _, cubeOrn = p.getBasePositionAndOrientation(self.botId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        # could also be pi/2 - abs(cubeEuler[0])
        return (1 - abs(cubeEuler[0])) * 0.1 -  abs(self.vt - self.vd) * 0.01

    def compute_done(self):
        # Check centre of mass of robot if is below 15 cm. If so, robot has fallen
        # over and episode shall be restarted
        # Check also if step counter exceed duration of each episode. If so, end
        # episode and restart.
        cubePos, _ = p.getBasePositionAndOrientation(self.botId)
        return cubePos[2] < 0.15 or self.envStepCounter >= 1500

