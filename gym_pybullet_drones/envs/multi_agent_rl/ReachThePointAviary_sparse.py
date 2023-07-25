import os
import random

import numpy as np
import pybullet as p
from gym import spaces

from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.enums import DroneModel, Physics

WALL_COLLISION_MALUS = -5
SPHERE_COLLISION_MALUS = -2
DRONE_RADIUS = .07
WORLDS_MARGIN = [-20, 60, -10, 10, 0, 10]  # minX maxX minY maxY minZ maxZ
WORLDS_MARGIN_MINUS_DRONE_RADIUS = WORLDS_MARGIN.copy()

NUMBER_OF_STACKED_OBS = 3

# world margin alredy take the radius of the drone
WORLDS_MARGIN_MINUS_DRONE_RADIUS[0] = WORLDS_MARGIN_MINUS_DRONE_RADIUS[0] + DRONE_RADIUS
WORLDS_MARGIN_MINUS_DRONE_RADIUS[1] = WORLDS_MARGIN_MINUS_DRONE_RADIUS[1] - DRONE_RADIUS
WORLDS_MARGIN_MINUS_DRONE_RADIUS[2] = WORLDS_MARGIN_MINUS_DRONE_RADIUS[2] + DRONE_RADIUS
WORLDS_MARGIN_MINUS_DRONE_RADIUS[3] = WORLDS_MARGIN_MINUS_DRONE_RADIUS[3] - DRONE_RADIUS
WORLDS_MARGIN_MINUS_DRONE_RADIUS[4] = WORLDS_MARGIN_MINUS_DRONE_RADIUS[4] + DRONE_RADIUS
WORLDS_MARGIN_MINUS_DRONE_RADIUS[5] = WORLDS_MARGIN_MINUS_DRONE_RADIUS[5] - DRONE_RADIUS

# threasold used to punish drone when it get near to spheres
SPHERES_THRESHOLD = 0.30
BOUNDARIES_THRESHOLD = 0.25
DIFFICULTY = 'medium'
# working dir need to be constant
WORKING_DIR = os.getcwd()


class ReachThePointAviary_sparse(BaseMultiagentAviary):
    """Multi-agent RL problem: leader-follower."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 2,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=np.vstack([np.array([0, 0]), \
                           np.array([0, 0]), \
                           np.ones(2)]).transpose().reshape(2, 3),
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 freq: int = 240,
                 aggregate_phy_steps: int = 5,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.episode = 0
        self.closest_sphere_distance = {}
        self.prev_sphere_treshold = 0
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        self.EPISODE_LEN_SEC = 500
        self.last_drones_dist = [1000000 for _ in self.get_agent_ids()]
        self.done_ep = {i: False for i in self.get_agent_ids()}
        self.prev_x_drones_pos = {int(i): self.INIT_XYZS[i, 0] for i in range(self.NUM_DRONES)}
        self.actual_step_drones_states = np.array([], dtype=np.float64)
        self.drone_has_collided = {i: (False, [0, 0, 0]) for i in range(self.NUM_DRONES)}
        self.drone_has_won = {i: False for i in range(self.NUM_DRONES)}

        self.drone_stacked_obs = {i: np.array([1 for _ in range(3 * 5)], dtype=np.float64)
                                  for i in
                                  range(self.NUM_DRONES)}
        self.drone_subgoal_counter = [0,0]
        self.subgoal_position = [0,5,10,15,20,25,30,35,40,45,50,55,60]

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.
        This method is called once per reset, the environment is recreated each time, maybe caching sphere is a good idea(Gyordan)
        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        import csv
        import assignment1 as module_path
        from random import randrange
        p.setAdditionalSearchPath(
            WORKING_DIR + "/shapes/")  # used by loadURDF
        # Override of small_sphere.urdf for changing starting radius to match collision
        # disable shadow for performance
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        n_env = 100
        difficulty = "/" + DIFFICULTY + "/"
        if self.episode % 20 == 0:
            env_number = str(randrange(n_env))
            print('CHOOSEN_ENV' + env_number)
            csv_file_path = os.path.dirname(
                module_path.__file__) + "/environment_generator/generated_envs" + difficulty + "{0}/static_obstacles.csv".format(
                "environment_" + env_number)

            with open(csv_file_path, mode='r') as infile:
                reader = csv.reader(infile)
                self.spheres = [[str(rows[0]), float(rows[1]), float(rows[2]), float(rows[3]), float(rows[4])] for rows
                                in
                                reader]
        # adding one spheres in front of the drones to avoid lucky wins
        self.spheres.append(['sphere_small.urdf', self.INIT_XYZS[0][0] + random.uniform(4, 8),
                             self.INIT_XYZS[0][1] + random.uniform(0, 0.15),
                             self.INIT_XYZS[0][2] + random.uniform(0, 0.15),
                             random.uniform(0.30, 1.25)])
        self.spheres.append(['sphere_small.urdf', self.INIT_XYZS[1][0] + random.uniform(4, 8),
                             self.INIT_XYZS[1][1] + random.uniform(0, 0.15),
                             self.INIT_XYZS[1][2] + random.uniform(0, 0.15),
                             random.uniform(0.30, 1.25)])

        for sphere in self.spheres:
            temp = p.loadURDF(sphere[0],
                              sphere[1:4:],
                              p.getQuaternionFromEuler([0, 0, 0]),
                              physicsClientId=self.CLIENT,
                              useFixedBase=True,
                              globalScaling=1 * (sphere[4]),

                              # flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                              )
            p.changeVisualShape(temp, -1, rgbaColor=[0, 0, 1, 1])

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.
        Returns
        -------
        dict[int, ndarray]
            A Dict() of Box() of size 1, 3, or 3, depending on the action type,
            indexed by drone Id in integer format.
        """
        if self.ACT_TYPE in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
            act_lower_bound = np.array([-1, -1])
            act_upper_bound = np.array([1, 1])
            return spaces.Dict({i: spaces.MultiDiscrete([3, 3]) for i in range(self.NUM_DRONES)})

        elif self.ACT_TYPE == ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
            size = 1
        else:
            print("[ERROR] in BaseMultiagentAviary._actionSpace()")
            exit()
        return spaces.Dict({i: spaces.Box(low=-1 * np.ones(size),
                                          high=np.ones(size),
                                          dtype=np.float64
                                          ) for i in range(self.NUM_DRONES)})

    ################################################################################
    def step(self,
             action
             ):
        # this is done, because step use _computeXXX methods
        self.actual_step_drones_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)],
                                                  dtype=np.float64)
        new_action = {}
        for k, v in action.items():
            new_action[k] = np.array([1, v[0] - 1, v[1] - 1, 1], dtype=np.float64)
        # x = {0: np.array([0.00001, 0, 0, 1]), 1: np.array([0.1, 0, 0, 1])}
        # if self.step_counter >= 100 and self.step_counter <= 600:
        #    x = {0: np.array([0.00001, 1, 0, 1]), 1: np.array([0.1, 0, 0, 1])}
        # if self.step_counter >= 1000:
        #    print(self.EPISODE_LEN_SEC)
        # print(self.prev_drones_pos[0])
        return super().step(new_action)
    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone.

        """
        rewards = {}
        for i in self.get_agent_ids():
            # if self.drone_has_collided[i]:
            #    rewards[i] = 0
            #    continue
            self.obs[i]
            #punishment_near_spheres = self.negRewardBaseOnSphereDistance(i)
            pushishment_near_walls = self.negRewardBaseOnTouchBoundary(i)
            rewards[i] = 0
            # drone has won
            if self.actual_step_drones_states[i, 0] >= WORLDS_MARGIN[1] -0.5:
                self.drone_has_collided[i] = (True, self.actual_step_drones_states[i, 0:3])
                rewards[i] = 500
                self.drone_has_won[i] = True
            elif pushishment_near_walls < 0:
                rewards[i] += pushishment_near_walls

            elif self.actual_step_drones_states[i, 0] > self.prev_x_drones_pos[i] and self.actual_step_drones_states[i,0] > self.subgoal_position[self.drone_subgoal_counter[i]]:
                rewards[i] = 10
                self.drone_subgoal_counter[i] +=1


            if self.actual_step_drones_states[i, 0] > self.prev_x_drones_pos[i]:
                self.prev_x_drones_pos[i] = self.actual_step_drones_states[i, 0]

        return rewards
    ################################################################################

    def _computeReward_old2(self):
        """Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone.

        """
        rewards = {}
        for i in self.get_agent_ids():
            # if self.drone_has_collided[i]:
            #    rewards[i] = 0
            #    continue
            self.obs[i]
            punishment_near_spheres = self.negRewardBaseOnSphereDistance(i)
            pushishment_near_walls = self.negRewardBaseOnTouchBoundary(i)
            rewards[i] = 0
            # drone has won
            if self.actual_step_drones_states[i, 0] >= WORLDS_MARGIN[1] -0.5:
                self.drone_has_collided[i] = (True, self.actual_step_drones_states[i, 0:3])
                rewards[i] = 500
                self.drone_has_won[i] = True
            elif pushishment_near_walls < 0:
                rewards[i] += pushishment_near_walls

            else:

                #rewards[i] += punishment_near_spheres
                rewards[i] += self.rewardBaseOnForward(self.actual_step_drones_states[i, :3],
                                                       self.prev_x_drones_pos[i],
                                                       self.actual_step_drones_states[i, 10])

            if self.actual_step_drones_states[i, 0] > self.prev_x_drones_pos[i]:
                self.prev_x_drones_pos[i] = self.actual_step_drones_states[i, 0]

        return rewards

    ################################################################################

    def _computeReward_old(self):
        """Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone.

        """
        rewards = {}
        for i in self.get_agent_ids():
            # if self.drone_has_collided[i]:
            #    rewards[i] = 0
            #    continue
            punishment_near_spheres = self.negRewardBaseOnSphereDistance(i)
            pushishment_near_walls = self.negRewardBaseOnTouchBoundary(i)

            rewards[i] = 0.002
            # drone has won
            if self.actual_step_drones_states[i, 0] >= WORLDS_MARGIN[1]:
                self.drone_has_collided[i] = (True, self.actual_step_drones_states[i, 0:3])
                rewards[i] = 50
                self.drone_has_won[i] = True
            else:
                # stare lontano dal muro y
                if abs(self.obs[i][4]) > 0.8:
                    rewards[i] -= 0.001
                # stare lontano dal muro z
                if 0.8 < self.obs[i][5] < 0.2:
                    rewards[i] -= 0.001
                # c'è almeno una sfera tra tutte allineata in y e z
                if self.obs[i][6:36:3].min() == 0 and self.obs[i][7:36:3].min() == 0:
                    rewards[i] -= 0.003

        return rewards

    ################################################################################
    def rewardBaseOnForward(self, drone_pos, prev_x_drone_pos, vel_x):
        """Compute the reward based on distance from x, if the drone approaches the target point (x=WORLDS_MARGIN[1]) the reward goes up
        """
        # return -0.5 * np.linalg.norm(np.array([WORLDS_MARGIN[1], drone_pos[1], drone_pos[2]]) - drone_pos)
        # Speed max is self.SPEED_LIMIT, see BaseMultiagentAviary, use min max scaling
        vel_x = vel_x if vel_x <= self.SPEED_LIMIT else self.SPEED_LIMIT
        if prev_x_drone_pos < drone_pos[0]:
            return self._minMaxScaling(vel_x, 0, self.SPEED_LIMIT)
        else:
            return 0

    ################################################################################
    def negRewardBaseOnTouchBoundary(self, drone_id):
        """ If the drones touch world boundary receive a reward of -100
        """
        neg_rew, collided = self.boundaries_incremental_punishment(self.actual_step_drones_states[drone_id, 0:3])
        if collided:
            self.drone_has_collided[drone_id] = (True, self.actual_step_drones_states[drone_id, 0:3])
            return WALL_COLLISION_MALUS
        else:
            return neg_rew

    ################################################################################
    def negRewardBaseOnSphereDistance(self, drone_index):
        """ If the drones touch a sphere receive a reward of -100
        """
        self.closest_sphere_distance[drone_index] = self.getClosestSpheres(
            self.actual_step_drones_states[drone_index, 0:3])
        reward = 0
        for sphere in self.closest_sphere_distance[drone_index]:
            radius = sphere['radius']
            dist = sphere['dist'] - radius - DRONE_RADIUS
            if dist <= SPHERES_THRESHOLD:
                if dist <= 0:
                    # drone collide with a sphere
                    self.drone_has_collided[drone_index] = (True, self.actual_step_drones_states[drone_index, 0:3])
                    reward = SPHERE_COLLISION_MALUS
                    break
                # the more near the drone gets the more penalty it receive
                reward = -(2 * (SPHERES_THRESHOLD - dist))
                break
        return reward

    ################################################################################
    def reset(self):
        import random
        self.episode += 1
        self.drone_subgoal_counter = [0,0]
        self.drone_has_won = {i: False for i in range(self.NUM_DRONES)}
        # check WORLDS_MARGIN for this, randomize the spawn position remaining away from sphere spawn area, need to be done before
        # super.reset()
        # self.INIT_XYZS[0] = np.array([random.randrange(2, 3), random.randrange(0, 2), random.randrange(2, 8)])
        # self.INIT_XYZS[1] = np.array([random.randrange(2, 3), random.randrange(0, 1), random.randrange(2, 8)])
        self.INIT_XYZS = np.vstack([np.array([0, 0]), \
                           np.array([0, 0]), \
                           np.ones(2)]).transpose().reshape(2, 3)
        self.INIT_XYZS[0] = np.array([random.randrange(-6, -3), random.randrange(-5, 5), random.randrange(2, 8)])
        self.INIT_XYZS[1] = np.array([random.randrange(-6, -3), random.randrange(-5, 5), random.randrange(2, 8)])

        self.prev_x_drones_pos = {int(i): self.INIT_XYZS[i, 0] for i in range(self.NUM_DRONES)}
        self.drone_has_collided = {i: (False, [0, 0, 0]) for i in range(self.NUM_DRONES)}
        self.drone_stacked_obs = {i: np.array([1 for _ in range(3 * 5)], dtype=np.float64) for i in
                                  range(self.NUM_DRONES)}
        return super().reset()

    ################################################################################
    def getClosestSpheres(self, drone_pos):
        """
        get the first 10 closest spheres with sphere_x > drone_x
        Return: dictionary of x,y,z distance from drone_pos, sphere radius, x,y,z of sphere, and norm distance
        """
        import operator
        distances = []
        for sphere in self.spheres:
            sphere_x = sphere[1]
            drone_x = drone_pos[0]
            radius = sphere[4]
            if (sphere_x + radius + DRONE_RADIUS) >= drone_x:
                sphere_y = sphere[2]
                drone_y = drone_pos[1]
                sphere_z = sphere[3]
                drone_z = drone_pos[2]

                distances.append({"x_center_dist": sphere_x - drone_x,
                                  "y_center_dist": sphere_y - drone_y,
                                  "z_center_dist": sphere_z - drone_z,
                                  "radius": radius,
                                  "x_sphere_pos": sphere_x,
                                  "y_sphere_pos": sphere_y,
                                  "z_sphere_pos": sphere_z,
                                  "dist": np.linalg.norm(drone_pos - sphere[1:4:])})
        sorted_dist = sorted(distances, key=operator.itemgetter('dist'))

        # fix in the case that i have surpassed all the spheres, needed 10 otherwise it will crash
        while len(sorted_dist) < 10:
            sorted_dist.append(
                {'x_center_dist': 10000, 'y_center_dist': 10000, 'z_center_dist': 10000, 'radius': 0.1,
                 'x_sphere_pos': 10000, 'y_sphere_pos': 10000, 'z_sphere_pos': 10000,
                 'dist': 10000})

        return sorted_dist[:10]

    ################################################################################
    def hit_world(self, drone_xyz):
        """
        Return: True if the drones touch world boundary False otherwise
        NB:This method alredy take in consideration drone radius
        """

        MIN_X = WORLDS_MARGIN_MINUS_DRONE_RADIUS[0]
        # If reach max_x is won not lost
        # MAX_X = WORLDS_MARGIN_MINUS_DRONE_RADIUS[1]
        MIN_Y = WORLDS_MARGIN_MINUS_DRONE_RADIUS[2]
        MAX_Y = WORLDS_MARGIN_MINUS_DRONE_RADIUS[3]
        MIN_Z = WORLDS_MARGIN_MINUS_DRONE_RADIUS[4]
        MAX_Z = WORLDS_MARGIN_MINUS_DRONE_RADIUS[5]
        # or drone_xyz[0] >= MAX_X \
        if drone_xyz[0] <= MIN_X \
                or MIN_Y >= drone_xyz[1] \
                or drone_xyz[1] >= MAX_Y \
                or MIN_Z >= drone_xyz[2] \
                or drone_xyz[2] >= MAX_Z:
            return True
        return False

    ################################################################################
    def boundaries_incremental_punishment(self, drone_xyz):
        """
        Return: True if the drones touch world boundary False otherwise
        NB:This method alredy take in consideration drone radius
        """
        penalty = 0
        PENALTY_MULTIPLIER = 0.25
        MIN_X = WORLDS_MARGIN_MINUS_DRONE_RADIUS[0]
        # If reach max_x is won not lost
        # MAX_X = WORLDS_MARGIN_MINUS_DRONE_RADIUS[1]
        MIN_Y = WORLDS_MARGIN_MINUS_DRONE_RADIUS[2]
        MAX_Y = WORLDS_MARGIN_MINUS_DRONE_RADIUS[3]
        MIN_Z = WORLDS_MARGIN_MINUS_DRONE_RADIUS[4]
        MAX_Z = WORLDS_MARGIN_MINUS_DRONE_RADIUS[5]

        if drone_xyz[0] > MIN_X:
            if (drone_xyz[0] - MIN_X) <= BOUNDARIES_THRESHOLD:
                penalty -= (0.5 * (BOUNDARIES_THRESHOLD - (drone_xyz[0] - MIN_X)))
        else:
            return 0, True
        # NO MAX X CHECKED IN REWARD
        if drone_xyz[1] > MIN_Y:
            if (drone_xyz[1] - MIN_Y) <= BOUNDARIES_THRESHOLD:
                penalty -= (PENALTY_MULTIPLIER * (BOUNDARIES_THRESHOLD - (drone_xyz[1] - MIN_Y)))
        else:
            return 0, True
        if drone_xyz[1] < MAX_Y:
            if (MAX_Y - drone_xyz[1]) <= BOUNDARIES_THRESHOLD:
                penalty -= (PENALTY_MULTIPLIER * (BOUNDARIES_THRESHOLD - (MAX_Y - drone_xyz[1])))
        else:
            return 0, True
        if drone_xyz[2] > MIN_Z:
            if (drone_xyz[2] - MIN_Z) <= BOUNDARIES_THRESHOLD:
                penalty -= (PENALTY_MULTIPLIER * (BOUNDARIES_THRESHOLD - (drone_xyz[2] - MIN_Z)))
        else:
            return 0, True
        if drone_xyz[2] < MAX_Z:
            if (MAX_Z - drone_xyz[2]) <= BOUNDARIES_THRESHOLD:
                penalty -= (PENALTY_MULTIPLIER * (BOUNDARIES_THRESHOLD - (MAX_Z - drone_xyz[2])))
        else:
            return 0, True

        return penalty, False

    ################################################################################
    def _computeDone(self):
        """Computes the current done value(s).

        Returns
        -------
        dict[int | "__all__", bool]
            Dictionary with the done value of each drone and
            one additional boolean value for key "__all__".

        """

        local_dones = {}
        # Take only the alive drones and then using calculation from reward calculate dones
        for i in self.get_agent_ids().copy():
            if self.drone_has_collided[i][0]:
                local_dones[i] = True
                self.get_agent_ids().remove(i)
                continue
            else:
                local_dones[i] = False

        local_dones["__all__"] = (True if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC else False) or all(
            [local_dones[k] for k in self.get_agent_ids()])
        return local_dones

    ################################################################################

    def is_drone_hit_spheres(self, drone_index):
        for sphere in self.closest_sphere_distance[drone_index]:
            if sphere['dist'] - sphere['radius'] - DRONE_RADIUS <= 0:
                # self.step_drones_states = self._getDroneStateVector(drone_index)
                return True
        return False

    ################################################################################
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[int, dict[]]
            Dictionary of empty dictionaries.

        """

        info_dict = {i: {} for i in self.get_agent_ids()}
        for i in self.get_agent_ids():
            info_dict[i]["won"] = self.drone_has_won[i]
            info_dict[i]["pos"] = self.actual_step_drones_states[i, 0:3]
        return info_dict

    ################################################################################

    def _observationSpaceOld(self):
        from gym import spaces
        # x is from 0 to 1, because we don't care about behind spheres
        #             x_dist  y_dist  z_dist r   sphere
        sphere_low = [0, -1, -1, 0] * 10
        sphere_high = [1, 1, 1, 1] * 10
        #      X   Y  Z   R   P   Y   VX  VY  VZ  WX  WY  WZ  WBY WBZ
        low = [-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0] + sphere_low
        high = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + sphere_high

        return spaces.Dict({i: spaces.Box(low=np.array(low),
                                          high=np.array(high),
                                          dtype=np.float64
                                          ) for i in range(self.NUM_DRONES)})

    ################################################################################

    def _observationSpace(self):
        from gym import spaces
        # x is from 0 to 1, because we don't care about behind spheres
        #             y_aling z_aling dist   sphere
        sphere_low = [0, 0, 0] * 10
        sphere_high = [1, 1, 1] * 10
        #      X   Y  Z  NT  WBY WBZ
        low = [-1, -1, 0, 0, -1, 0] + sphere_low
        high = [1, 1, 1, 1, 1, 1] + sphere_high

        return spaces.Dict({i: spaces.Box(low=np.array(low),
                                          high=np.array(high),
                                          dtype=np.float64
                                          ) for i in range(self.NUM_DRONES)})

    ################################################################################
    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by Id in integer format,
            each a Box() os shape (H,W,4) or (12,) depending on the observation type.

        """
        ############################################################
        #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
        # return {   i   : self._clipAndNormalizeState(self._getDroneStateVector(i)) for i in range(self.NUM_DRONES) }
        ############################################################
        #### OBS SPACE OF SIZE 52
        obs_54 = np.zeros((self.NUM_DRONES, 36), dtype=np.float64)
        for i in self.get_agent_ids():
            drone_state = self._getDroneStateVector(i)
            obs = self._clipAndNormalizeState(drone_state)
            boundaries_distances = self.get_normalized_y_z_boundaries(drone_state[0:3])
            close_sphere = self.getClosestSpheres(drone_state[0:3])
            # normalize_sphere = self.clipAndNormalizeSphere_old(close_sphere) # if used need to change _observationSpace
            normalize_sphere = self.clipAndNormalizeSphere_obs_local(close_sphere)
            normalized_time = self._minMaxScaling(self.step_counter / self.SIM_FREQ, 0, self.EPISODE_LEN_SEC,
                                                  standard_rng=True)

            obs_54[i, :] = np.hstack(
                [obs[0:3], normalized_time, boundaries_distances,
                 np.append(normalize_sphere[:3 * 5], self.drone_stacked_obs[i])]).reshape(
                36, )

            self.drone_stacked_obs[i] = np.array(normalize_sphere[:3 * 5], dtype=np.float64)

        return {i: obs_54[i, :] for i in self.get_agent_ids()}

    #############################################################
    def get_normalized_y_z_boundaries(self, drone_pos):
        # minX maxX minY maxY minZ maxZ

        distances = [0, 0]
        # x is not used, because the drone need to go forward, can't go backward
        # y is normalized between -1 and 1

        distances[0] = self._minMaxScaling(
            drone_pos[1],
            WORLDS_MARGIN_MINUS_DRONE_RADIUS[2],
            WORLDS_MARGIN_MINUS_DRONE_RADIUS[3],
            False)
        # Z is normalized between 0 and 1, because z pos canno't be negative
        distances[1] = self._minMaxScaling(
            drone_pos[2],
            WORLDS_MARGIN_MINUS_DRONE_RADIUS[4],
            WORLDS_MARGIN_MINUS_DRONE_RADIUS[5])
        return distances

    #############################################################

    def clipAndNormalizeSphere_old(self, spheres):
        MAX_MARGIN = np.array([WORLDS_MARGIN[1], WORLDS_MARGIN[3], WORLDS_MARGIN[5]], dtype=np.float64)
        MIN_MARGIN = np.array([WORLDS_MARGIN[0], WORLDS_MARGIN[2], WORLDS_MARGIN[4]], dtype=np.float64)
        MIN_DISTANCE = 0

        MAX_DISTANCE = np.linalg.norm(MAX_MARGIN - MIN_MARGIN)
        norm_and_clipped = []
        for s in spheres:
            clipped_pos_x = np.clip(s['x'], WORLDS_MARGIN[0], WORLDS_MARGIN[1])
            clipped_pos_y = np.clip(s['y'], WORLDS_MARGIN[2], WORLDS_MARGIN[3])
            clipped_pos_z = np.clip(s['z'], WORLDS_MARGIN[4], WORLDS_MARGIN[5])

            normalized_x = clipped_pos_x / WORLDS_MARGIN[1]
            normalized_y = clipped_pos_y / WORLDS_MARGIN[3]
            normalized_z = clipped_pos_z / WORLDS_MARGIN[5]
            clipped_distance = np.clip(s["dist"] - s["radius"] - DRONE_RADIUS, MIN_DISTANCE, MAX_DISTANCE)
            normalized_dist = clipped_distance / MAX_DISTANCE
            norm_and_clipped.extend([normalized_x, normalized_y, normalized_z, normalized_dist])

        return norm_and_clipped

    ################################################################################
    # TODO Bugged? need review
    def clipAndNormalizeSphere_obs_global(self, spheres):
        MAX_MARGIN = np.array([WORLDS_MARGIN[1], WORLDS_MARGIN[3], WORLDS_MARGIN[5]], dtype=np.float64)
        MIN_MARGIN = np.array([WORLDS_MARGIN[0], WORLDS_MARGIN[2], WORLDS_MARGIN[4]], dtype=np.float64)
        MIN_DISTANCE = 0

        MAX_DISTANCE = np.linalg.norm(MAX_MARGIN - MIN_MARGIN) - DRONE_RADIUS
        norm_and_clipped = []

        max_dist_x = abs(WORLDS_MARGIN[0]) + abs(WORLDS_MARGIN[1])
        max_dist_y = abs(WORLDS_MARGIN[2]) + abs(WORLDS_MARGIN[3])
        max_dist_z = abs(WORLDS_MARGIN[4] + abs(WORLDS_MARGIN[5]))
        # if field is -20,60 the min dist per axis is -80 +80
        for s in spheres:
            clipped_pos_x = np.clip(s['x_dist'] - s["radius"] - DRONE_RADIUS,  # posizione
                                    -max_dist_x + s["radius"] + DRONE_RADIUS,  # min distanza con segno
                                    max_dist_x - s["radius"] - DRONE_RADIUS)  # max distanza con segno
            clipped_pos_y = np.clip(s['y_dist'] - s["radius"] - DRONE_RADIUS,
                                    -max_dist_y + s["radius"] + DRONE_RADIUS,
                                    max_dist_y - s["radius"] - DRONE_RADIUS)
            clipped_pos_z = np.clip(s['z_dist'] - s["radius"] - DRONE_RADIUS,
                                    -max_dist_z + s["radius"] + DRONE_RADIUS,
                                    max_dist_z - s["radius"] - DRONE_RADIUS)

            normalized_x = clipped_pos_x / (max_dist_x - s["radius"] - DRONE_RADIUS)
            normalized_y = clipped_pos_y / (max_dist_y - s["radius"] - DRONE_RADIUS)
            normalized_z = clipped_pos_z / (max_dist_z - s["radius"] - DRONE_RADIUS)
            clipped_distance = np.clip(s["dist"] - s["radius"] - DRONE_RADIUS, MIN_DISTANCE, MAX_DISTANCE - s["radius"])
            normalized_dist = clipped_distance / (MAX_DISTANCE - s["radius"])
            norm_and_clipped.extend([normalized_x, normalized_y, normalized_z, normalized_dist])

        return norm_and_clipped

    ################################################################################
    def clipAndNormalizeSphere_obs_local(self, spheres):
        norm_and_clipped = []
        threashold = 2
        # min dist, max dist 0, 5
        for s in spheres:
            if (s["dist"] - s["radius"] - DRONE_RADIUS) < threashold:
                normalized_x = self._minMaxScaling(s["x_center_dist"] - s["radius"] - DRONE_RADIUS, 0, threashold,
                                                   standard_rng=True)

                normalized_y = self._minMaxScaling(abs(s["y_center_dist"]) - s["radius"] - DRONE_RADIUS, 0, threashold,
                                                   standard_rng=True)
                normalized_z = self._minMaxScaling(abs(s["z_center_dist"]) - s["radius"] - DRONE_RADIUS, 0, threashold,
                                                   standard_rng=True)
                normalized_dist = self._minMaxScaling(s["dist"] - s["radius"] - DRONE_RADIUS, 0, threashold,
                                                      standard_rng=True)
            else:
                normalized_x = 1
                normalized_y = 1
                normalized_z = 1
                normalized_dist = 1

            norm_and_clipped.extend([normalized_y, normalized_z, normalized_dist])
        return norm_and_clipped

    ################################################################################
    def _minMaxScaling(self, val, min_v, max_v, standard_rng=True):
        if standard_rng:
            return max(0, min((val - min_v) / (max_v - min_v), 1))
        else:
            return max(-1, min(2 * (val - min_v) / (
                    max_v - min_v) - 1, 1))

    ################################################################################
    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        # Check BaseMultiagentAviary to increase the maximum speed
        MAX_LIN_VEL_XY = self.SPEED_LIMIT
        MAX_LIN_VEL_Z = self.SPEED_LIMIT
        MIN_X = -6
        MAX_X = WORLDS_MARGIN[1]
        MIN_Y = WORLDS_MARGIN[2]
        MAX_Y = WORLDS_MARGIN[3]
        MIN_Z = WORLDS_MARGIN[4]
        MAX_Z = WORLDS_MARGIN[5]

        MAX_PITCH_ROLL = np.pi  # Full range

        pos_x = np.clip(state[0], MIN_X, MAX_X)
        pos_y = np.clip(state[1], MIN_Y, MAX_Y)
        pos_z = np.clip(state[2], MIN_Z, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               pos_x,
                                               pos_y,
                                               pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_x = self._minMaxScaling(pos_x, MIN_X, MAX_X, False)
        normalized_pos_y = self._minMaxScaling(pos_y, MIN_Y, MAX_Y, False)
        normalized_pos_z = self._minMaxScaling(pos_z, MIN_Z, MAX_Z, True)
        normalized_rp = [self._minMaxScaling(val, -MAX_PITCH_ROLL, MAX_PITCH_ROLL, False) for val in clipped_rp]
        # self._minMaxScaling(clipped_rp, -MAX_PITCH_ROLL, MAX_PITCH_ROLL, False)
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = [self._minMaxScaling(val, -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY, False) for val in clipped_vel_xy]
        # normalized_vel_xy = self._minMaxScaling(clipped_vel_xy, -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY, False)
        normalized_vel_z = self._minMaxScaling(clipped_vel_z, -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z, False)

        normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(
            state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_x,
                                      normalized_pos_y,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20, )

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_x,
                                      clipped_pos_y,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.

        """
        if not (clipped_pos_x == np.array(state[0])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped x position {:.2f}".format(
                      state[0]))
        if not (clipped_pos_y == np.array(state[1])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped y position {:.2f}".format(
                      state[1]))
        if not (clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not (clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(
                      state[7], state[8]))
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(
                      state[10], state[11]))
        if not (clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
