# Code written by ME5418 AY23/24 Group 18
# This python define a class to create a gym environment (PyBullet Real-time Physics Simulation with Aliengo Robot) for our project.

import gym
import pybullet
from pybullet_utils import bullet_client
import numpy as np
import os
import random
from DDPGNet import scale_actions

# Initialization values for reward (Tentative values, might change in future).

ROBOT_DEVIATION_PENALTY = -5
ROBOT_HEIGHT_PENALTY = -5
ROBOT_VEL_REWARD = +20
ROBOT_PITCH_PENALTY = -5
ROBOT_LARGE_ANGLE_PENALTY = -2

ROBOT_TORSO_HEIGHT_THRES = 0.5
ROBOT_TORSO_FRONT_VEL_THRES = 1.5
ROBOT_TORSO_SIDE_VEL_THRES = 0.2
ROBOT_TORSO_ORI_THRES = 30
ROBOT_JOINT_ANGLE_THRES = 120

# This value below will affect how long each action step took inside the PyBullet simulation. This way we can discretize time.
ACTION_REPEAT = 10

class AliengoEnv(gym.Env):

    # Initialise env
    def __init__(self, render = False, Policy = 1):
        # Launch Pybullet and initialize attributes
        if render:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI) # This would launch PyBullet Simulation with visualization (PyBullet window will pop out)
        else:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT) # This would launch PyBullet Simulation without visualization (PyBullet Window hidden)
        self.script_directory = os.path.dirname(os.path.abspath(__file__)) # get path to this script. So that we could locate urdf files easily when importing models.
        self.state = {}
        self.observation = {}
        self.action_boundary = {}
        self.action = ()
        self.robot = None # aliengo's ID used in pybullet
        self.floor = None # floor's ID used in pybullet
        self.robot_torso_position = None
        self.robot_torso_velocity = None
        self.robot_torso_orientation = None
        self.each_joint_angle = None
        self.any_collision_between_each_link = None # True if there's collision
        self.finished = None # True if robot is in termination state
        self.joint_dict = {} # relate the joint ID used in pybullet Simulation to the joint name
        self.action2joint_idx_dict = {} # relate the index of the action tuple to the joint ID used in pybullet Simulation
        self.num_joints = None
        self.policy = Policy

    def reset(self, reload_urdf = False):
        # This method will reset the environment.

        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 0) # Start the configuration of PyBulet visualisation

        if reload_urdf: # If true, then we reload all model into pybullet
            gravity = -9.8
            if self.policy == 3:
                gravity = 0
            self._pybullet_client.setGravity(0, 0, gravity) # Set PyBullet Gravity
            self.robot = self._pybullet_client.loadURDF(self.script_directory + "/pybullet_data/aliengo/aliengo.urdf",[0.0164, 0.0079, 0.5257], self._pybullet_client.getQuaternionFromEuler([0,0,0])) # we import aliengo robot model using pybullet built-in function
            # self.floor = self._pybullet_client.createPlane(normalVector=[0, 0, 1])
            self.floor  = self._pybullet_client.loadURDF(self.script_directory + "/pybullet_data/plane/plane.urdf")
            # self.floor = self._pybullet_client.loadURDF(self.script_directory + "/pybullet_data/plane/plane.urdf", [0, 0, 0], self._pybullet_client.getQuaternionFromEuler([0,0,0])) # we also need to import ground floor using pybullet built-in function for robot to interact with
            self._pybullet_client.resetBasePositionAndOrientation(self.robot, [0.0164, 0.0079, 0.5257], self._pybullet_client.getQuaternionFromEuler([0,0,0])) # We use pybullet built-in function to reset robot position + orientation
            self._pybullet_client.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0]) # We use pybullet built-in function to reset robot velocity(linear & angular)
            self._pybullet_client.setTimeStep(0.002) # Set simulation timestep using pybullet built-in function
            self.get_robot_joint_info() # Update robot joint information
        else: # If false, we do not need to import the robot model again to save time.
            self._pybullet_client.resetBasePositionAndOrientation(self.robot, [0.0164, 0.0079, 0.5257], self._pybullet_client.getQuaternionFromEuler([0,0,0]))
            self._pybullet_client.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0]) #
            for i in range(self.num_joints):
                self._pybullet_client.resetJointState(self.robot,i,targetValue=0,targetVelocity=0) # We reset each joint to it's initial position, orientaion, velocity
        self._pybullet_client.resetDebugVisualizerCamera(cameraDistance = 2.0, cameraYaw = 0, cameraPitch = -30, cameraTargetPosition = [0, 0, 0]) # Here we use pybullet built-in function to set the camera position to see the robot from a nice position
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 1) # End the configuration of PyBullet visualisation
        self.get_observation() # Update robot observation
        self.get_action_boundary() # Update action boundary
        self.get_random_action() # Update actions
        return self.get_state() # Update and return robot state

    def get_robot_joint_info(self):
        # This method will:
        # One, relate the joint ID used in PyBullet Simulation to the joint name via the self.joint_dict.
        # Two, relate the index of the action tuple to the joint ID used in PyBullet Simulation via self.action2joint_idx_dict
        self.joint_dict.clear()
        self.num_joints = self._pybullet_client.getNumJoints(self.robot)
        for i in range(self.num_joints): # Loop through all joint inside PyBullet and get their information.
            joint_info = self._pybullet_client.getJointInfo(self.robot, i)
            joint_name = joint_info[1].decode("utf-8")
            self.joint_dict[i]=joint_name
        self.action2joint_idx_dict.clear()
        action_list = ['FR_upper_joint', 'FR_lower_joint', 'FL_upper_joint', 'FL_lower_joint', 'RR_upper_joint', 'RR_lower_joint', 'RL_upper_joint', 'RL_lower_joint']
        self.action2joint_idx_dict = {action_list.index(value): key for key, value in self.joint_dict.items() if value in action_list}
        # In usual case, after running these:
        # self.joint_dict = {0:"imu_joint", 1:"FR_hip_joint", 2:"FR_upper_joint", 3:"FR_lower_joint", 4:"FR_toe_fixed", 5:"FL_hip_joint", 6:"FL_upper_joint", 7:"FL_lower_joint", 8:"FL_toe_fixed", 9:"RR_hip_joint", 10:"RR_upper_joint", 11:"RR_lower_joint", 12:"RR_toe_fixed", 13:"RL_hip_joint", 14:"RL_upper_joint", 15:"RL_lower_joint", 16:"RL_toe_fixed"}
        # self.action2joint_idx_dict = {0: 2, 1: 3, 2: 6, 3: 7, 4: 10, 5: 11, 6: 14, 7: 15}
        return

    def get_observation(self):
        # Since observation information is included in state information. We extract information directly from state
        self.state = self.get_state()
        self.observation["robot_torso_position"] = self.state["robot_torso_position"]
        self.observation["robot_torso_velocity"] = self.state["robot_torso_velocity"]
        self.observation["robot_torso_orientation"] = self.state["robot_torso_orientation"]
        self.observation["robot_torso_ori_rate"] = self.state["robot_torso_ori_rate"]
        self.observation["each_joint_angle"] = self.state["each_joint_angle"]
        self.observation["each_joint_vel"] = self.state["each_joint_vel"]
        self.observation["leg_contact_with_ground"] = self.state["leg_contact_with_ground"]
        return self.observation

    def get_state(self):
        # This method will return the state of the robot
        robot_torso_position = (0, 0, 0) # px, py, pz values w.r.t. simulated world's origin, continuous
        robot_torso_velocity = (0, 0, 0) # vx, vy, vz values w.r.t. simulated world's origin, continuous
        robot_torso_ori_rate = (0, 0, 0)
        robot_torso_orientation = (0, 0, 0, 0) # quaternion values w.r.t. simulated world's origin, continuous
        each_joint_angle = {} # eight joints' angle, continuous
        each_joint_vel = {} # eight joints' velocity

        robot_torso_position, robot_torso_orientation = self._pybullet_client.getBasePositionAndOrientation(self.robot) # We use pybullet built-in function to obtain robot torso (parent link as described in aliengo urdf file) position and orientation
        robot_torso_velocity, robot_torso_ori_rate = self._pybullet_client.getBaseVelocity(self.robot) # We use pybullet built-in function to obtain robot torso (parent link as described in aliengo urdf file) velocity. [0] is to obtain linear velocity only.

        for i in range(self.num_joints): # Loop through every joint inside PyBullet and keep only the values of joints that we are interested
            joint_state = self._pybullet_client.getJointState(self.robot,i)
            joint_position = joint_state[0]
            joint_vel = joint_state[1]
            if self.joint_dict[i] in ["FR_upper_joint", "FR_lower_joint", "FL_upper_joint", "FL_lower_joint","RR_upper_joint","RR_lower_joint", "RL_upper_joint","RL_lower_joint"]:
                each_joint_angle[self.joint_dict[i]]=joint_position
                each_joint_vel[self.joint_dict[i]]=joint_vel

        # Update state information
        self.state["robot_torso_position"] = robot_torso_position
        self.state["robot_torso_velocity"] = robot_torso_velocity
        self.state["robot_torso_orientation"] = robot_torso_orientation
        self.state["robot_torso_ori_rate"] = robot_torso_ori_rate
        self.state["each_joint_angle"] = each_joint_angle
        self.state["each_joint_vel"] = each_joint_vel
        self.state["leg_contact_with_ground"] = self.get_leg_contact_with_ground()
        self.state["is_terminate"] = self.is_terminate(robot_torso_position)

        return self.state

    def single_time_step(self):
        self._pybullet_client.stepSimulation() # This line here will prompt Pybullet to process the timestep.
        return 0

    def get_leg_contact_with_ground(self):
        # return self._pybullet_client.getContactPoints(self.robot, self.floor)
        output = {}
        for linkidx in [("FR_toe", 4), ("FL_toe", 8), ("RR_toe", 12), ("RL_toe", 16)]:
            if len(self._pybullet_client.getContactPoints(self.robot, self.floor, linkIndexA = linkidx[1]) ) >0:
                output[linkidx[0]] = 1
            else:
                output[linkidx[0]] = 0
        # return [self._pybullet_client.getContactPoints(self.robot, self.floor, linkIndexA = 4), self._pybullet_client.getContactPoints(self.robot, self.floor, linkIndexA = 8), self._pybullet_client.getContactPoints(self.robot, self.floor, linkIndexA = 12), self._pybullet_client.getContactPoints(self.robot, self.floor, linkIndexA = 16)]
        return output

    def get_robot_collision(self):
        # This method to check for any collision between any two robot link. We iteratively check for any contact point between each and every link.
        collided = 0
        for link_index in range(self.num_joints):
            for link_index2 in range(self.num_joints):
                if abs(link_index-link_index2) > 1:
                    cont_pts = self._pybullet_client.getClosestPoints(self.robot, self.robot, linkIndexA=link_index, linkIndexB=link_index2, distance=0) # We use pybullet's built-in function to check for contact point between the two links.
                    if len(cont_pts) > 0:
                        collided = 1 # If the two link has contact point, then there is collision.
        return collided

    def get_action_boundary(self):
        self.action_boundary["high"] = 26.5
        self.action_boundary["low"] = -26.5
        self.action_boundary["size"] = 8
        return self.action_boundary # {"high" : 26.5, "low": -26.5, "size":8}

    def get_random_action(self):
        self.get_action_boundary()
        self.action = tuple(random.uniform(self.action_boundary["high"], self.action_boundary["low"]) for _ in range(self.action_boundary["size"])) # randomly get action value (maximum torque allowed) for all 8 joints.
        return self.action

    def step(self, action, step_count):
        if step_count == 0:
            self.action = action
        previous_action = self.action
        # This method will run an action, and return previous state, action taken, new state, reward, is it termination state?
        previous_state = self.state
        self.act(action)
        state = self.get_state()
        self.finished = state["is_terminate"]
        reward = self.get_reward(step_count, self.action)
        self.action = action
        return previous_state, action, state, reward, self.finished, previous_action

    def act(self, action):
        # This method will apply an action to the robot. We use Pybullet built-in function setJoinMotorControl2() for that.
        for _ in range(ACTION_REPEAT): # since PyBullet simulation's timestep can be very small, we repeteadly apply the control for a few times in order to make our discretize timestep longer.
            for i in range(len(action)): # Go through each joint to apply the velocity control with the maximum torque values.
                joint_idx = self.action2joint_idx_dict[i] # Get joint ID
                if(action[i]<0): # sign of control values determine the velocity direction.
                    self._pybullet_client.setJointMotorControl2(self.robot,joint_idx,self._pybullet_client.VELOCITY_CONTROL,targetVelocity=-1,force=abs(action[i]))
                else:
                    self._pybullet_client.setJointMotorControl2(self.robot,joint_idx,self._pybullet_client.VELOCITY_CONTROL,targetVelocity=1,force=abs(action[i]))
            self._pybullet_client.stepSimulation() # This line here will prompt Pybullet to process the timestep.
        return

    def is_terminate(self, pos):
        output = False
        if self.policy == 1: # for V1 and V2 case
            output = self.get_robot_collision()
            if pos[2] < 0.2:
                output = True
            if abs(pos[1]) > 1.5:
                output = True
        else:
            if abs(pos[1])>1.5:
                output = True
        return output

    def get_reward(self, step_count, action):
        # This method will calculate the total reward for the current state (after applying action)
        state = self.get_state()
        robot_torso_position = state["robot_torso_position"] #tuple (x, y, z)
        robot_torso_velocity = state["robot_torso_velocity"] #tuple (x, y, z)
        robot_torso_orientation = state["robot_torso_orientation"] #tuple (rx, ry, rz, rw)
        total_reward = 0

        if self.policy == 1: # for V1 case

            action_sum = 0
            for i in action:
                action_sum += i**2
            total_reward -= action_sum * 0.02 # large action penalty
            pitch = self._pybullet_client.getEulerFromQuaternion(robot_torso_orientation)[1]
            total_reward -= 20 * (pitch ** 2) # pitch penalty
            total_reward += 5 * robot_torso_velocity[0] # velocity reward
            total_reward -= 20 * ((robot_torso_position[2] - 0.45) ** 2) # height penalty
            if step_count < 15:
                pass
            elif step_count >=15 and step_count <= 30:
                total_reward += 10 * (step_count - 15)/15
            elif step_count > 30 and step_count <= 50:
                total_reward += 20 * (step_count - 30)/20
            else:
                total_reward += 25

        else: # for V3 case
            total_reward += 5 * robot_torso_velocity[0]
            total_reward -= 1.5 * abs(robot_torso_position[1])

        return total_reward

    def render(self):
        # This method will capture a snapshot of the PyBullet Simulation.
        # This is done by using pybullet's built in function of getCameraImage(), which allows us to get pixel value
        # We will provide argument related to the camera view and projection information.
        # Here we let the camera to track the robot position.
        base_pos = self.state["robot_torso_position"]
        _, _, px, _, _ = self._pybullet_client.getCameraImage(width = 480,
                                                              height = 360,
                                                              viewMatrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(cameraTargetPosition = base_pos, distance = 2.0, yaw = 0, pitch = -30, roll = 0, upAxisIndex = 2),
                                                              projectionMatrix = self._pybullet_client.computeProjectionMatrixFOV(fov=60, aspect=float(480) / 360, nearVal=0.1, farVal=100.0),
                                                              renderer = self._pybullet_client.ER_BULLET_HARDWARE_OPENGL)
        RGB_array = np.array(px)[:, :, :3] # Convert the pixel values from PyBUllet into rgb_array format
        return RGB_array

    def disconnect(self):
        # This method will properly terminate PyBullet Simulation.
        self._pybullet_client.resetSimulation() # Remove everything inside the simulation
        self._pybullet_client.disconnect()
        return
