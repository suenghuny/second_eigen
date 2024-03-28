#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Tushar Jain, Amanpreet Singh

Simulate a predator prey environment.
Each agent can just observe itself (it's own identity) i.e. s_j = j and vision sqaure around it.

Design Decisions:
    - Memory cheaper than time (compute)
    - Using Vocab for class of box:
         -1 out of bound,
         indexing for predator agent (from 2?)
         ??? for prey agent (1 for fixed case, for now)
    - Action Space & Observation Space are according to an agent
    - Rewards -0.05 at each time step till the time
    - Episode never ends
    - Obs. State: Vocab of 1-hot < predator, preys & units >
"""

# core modules
import random
import math
import curses

# 3rd party modules
import gym
import numpy as np
from gym import spaces

def get_distance(predator_position, prey_position):
    distance_x = predator_position[0]-prey_position[0]
    distance_y = predator_position[1] - prey_position[1]
    distance = (distance_x**2+distance_y**2)**0.5
    return distance



class PredatorPreyEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self,):
        self.__version__ = "0.0.1"

        # TODO: better config handling
        self.OUTSIDE_CLASS = 1
        self.PREY_CLASS = 2
        self.PREDATOR_CLASS = 3
        self.TIMESTEP_PENALTY = -0.05
        self.PREY_REWARD = 0
        self.POS_PREY_REWARD = 0.05
        self.episode_over = False

    def init_curses(self):
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_RED, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_CYAN, -1)
        curses.init_pair(4, curses.COLOR_GREEN, -1)


    def init_args(self, parser):
        env = parser.add_argument_group('Prey Predator task')





    def multi_agent_init(self, args):

        # General variables defining the environment : CONFIG
        params = ['dim', 'vision', 'moving_prey', 'mode', 'enemy_comm']
        for key in params:
            setattr(self, key, getattr(args, key))

        self.nprey = args.nenemies
        self.npredator = args.nfriendly
        self.dims = dims = (self.dim, self.dim)
        self.stay = not args.no_stay

        # if args.moving_prey:
        #     raise NotImplementedError
        #     # TODO

        # (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT, 4: STAY)
        # Define what an agent can do -
        if self.stay:
            self.naction = 5
        else:
            self.naction = 4

        self.action_space = spaces.MultiDiscrete([self.naction])

        self.BASE = (dims[0] * dims[1])
        self.OUTSIDE_CLASS  += self.BASE
        self.PREY_CLASS     += self.BASE
        self.PREDATOR_CLASS += self.BASE

        # Setting max vocab size for 1-hot encoding
        self.vocab_size = 1 + 1 + self.BASE + 1 + 1
        #          predator + prey + grid + outside

        # Observation for each agent will be vision * vision ndarray
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.vocab_size, (2 * self.vision) + 1, (2 * self.vision) + 1), dtype=int)
        # Actual observation will be of the shape 1 * npredator * (2v+1) * (2v+1) * vocab_size

        return

    def step(self, action):
        """
        The agents take a step in the environment.

        Parameters
        ----------
        action : list/ndarray of length m, containing the indexes of what lever each 'm' chosen agents pulled.

        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (object) :

            reward (float) : Ratio of Number of discrete levers pulled to total number of levers.
            episode_over (bool) : Will be true as episode length is 1
            info (dict) : diagnostic information useful for debugging.
        """
        if self.episode_over:
            raise RuntimeError("Episode is done")

        action = np.array(action).squeeze()
        action = np.atleast_1d(action)

        #np.random.randint(0, 5)
        for i, a in enumerate(action):
            self._take_action(i, a
                              )
        prey_action = np.random.randint(0, 5, self.nprey)
        for i, a in enumerate(prey_action):
            self._prey_action(i, a)

        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."


        self.episode_over = False
        #self.obs, visualize_positional_info, positional_info, observation_matrix = self._get_obs()

        debug = {'predator_locs':self.predator_loc,'prey_locs':self.prey_loc}
        return self._get_reward(), self.episode_over, debug#, visualize_positional_info, positional_info, observation_matrix, self.obs,

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.episode_over = False
        self.reached_prey = np.zeros(self.npredator)

        # Locations
        locs = self._get_cordinates()
        self.predator_loc, self.prey_loc = locs[:self.npredator], locs[self.npredator:]

        self._set_grid()

        # stat - like success ratio
        self.stat = dict()

        # Observation will be npredator * vision * vision ndarray
        self.obs = self._get_obs()
        return self.obs

    def seed(self):
        return

    def _get_cordinates(self):
        idx = np.random.choice(np.prod(self.dims),(self.npredator + self.nprey), replace=False)
        return np.vstack(np.unravel_index(idx, self.dims)).T

    def _set_grid(self):
        self.grid = np.arange(self.BASE).reshape(self.dims)
        # Mark agents in grid
        # self.grid[self.predator_loc[:,0], self.predator_loc[:,1]] = self.predator_ids
        # self.grid[self.prey_loc[:,0], self.prey_loc[:,1]] = self.prey_ids

        # Padding for vision
        self.grid = np.pad(self.grid, self.vision, 'constant', constant_values = self.OUTSIDE_CLASS)

        self.empty_bool_base_grid = self._onehot_initialization(self.grid)

    def get_action_feature(self):
        action_feature = np.eye(5, dtype=np.float)
        return action_feature


    def _get_obs(self):

        self.bool_base_grid = self.empty_bool_base_grid.copy()
        # (grid_dim+4, grid_dim+4, 629) 짜리 empty array 행렬 생성


        #print(self.bool_base_grid.shape)
        visualize_positional_info = list()
        positional_info = list()
        observation_matrix = [[], []]
        for i, p in enumerate(self.predator_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREDATOR_CLASS] += 1
            visualize_positional_info.append((p[0]*50, p[1]*50, 1))
            positional_info.append((p[0] /self.dims[0], p[1] /self.dims[0], 1))
            p_predator = p
            len_predator = len(self.predator_loc)
            for j, p_prey in enumerate(self.prey_loc):
                d = get_distance(p_predator, p_prey)
                if d <= self.vision:
                    observation_matrix[0].append(i)
                    observation_matrix[1].append(len_predator+j)
                    observation_matrix[0].append(len_predator+j)
                    observation_matrix[1].append(i)




        for i, p in enumerate(self.prey_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREY_CLASS] += 1
            visualize_positional_info.append((p[0]*10, p[1]*10 , 0))
            positional_info.append((p[0] / self.dims[0], p[1] / self.dims[0], 1))

        obs = []
        for p in self.predator_loc:
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            obs.append(self.bool_base_grid[slice_y, slice_x])

        if self.enemy_comm:
            for p in self.prey_loc:
                slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
                slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
                obs.append(self.bool_base_grid[slice_y, slice_x])

        obs = np.stack(obs)
        return obs, visualize_positional_info, positional_info, observation_matrix

    def get_avail_actions(self):
        avail_actions = []
        for idx in range(len(self.prey_loc)):
            avail_action = []
            for act in range(self.naction):
                if act == 4:
                    avail_action.append(True)

                # UP
                if act == 0:
                    if self.grid[max(0, self.prey_loc[idx][0] + self.vision - 1),self.prey_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
                        avail_action.append(True)
                    else:
                        avail_action.append(False)

                if act == 1:
                    if self.grid[self.prey_loc[idx][0] + self.vision,min(self.dims[1] - 1,self.prey_loc[idx][1] + self.vision + 1)] != self.OUTSIDE_CLASS:
                        avail_action.append(True)
                    else:
                        avail_action.append(False)

                # DOWN
                if act == 2:
                    if self.grid[min(self.dims[0] - 1,
                                                self.prey_loc[idx][0] + self.vision + 1),
                                            self.prey_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
                        avail_action.append(True)
                    else:
                        avail_action.append(False)


                # LEFT
                if act == 3:
                    if self.grid[self.prey_loc[idx][0] + self.vision,
                                            max(0,
                                                self.prey_loc[idx][1] + self.vision - 1)] != self.OUTSIDE_CLASS:
                        avail_action.append(True)
                    else:
                        avail_action.append(False)
            avail_actions.append(avail_action)
        return avail_actions

    def get_heterogeneous_graph(self):
        self.obs, visualize_positional_info, positional_info, observation_matrix = self._get_obs()
        return positional_info, observation_matrix, 1,visualize_positional_info, 1

    def _prey_action(self, idx, act):
        if act == 4:
            return

        # UP
        if act == 0 and self.grid[max(0, self.prey_loc[idx][0] + self.vision - 1),
                                  self.prey_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.prey_loc[idx][0] = max(0, self.prey_loc[idx][0] - 1)

        # RIGHT
        elif act == 1 and self.grid[self.prey_loc[idx][0] + self.vision,
                                    min(self.dims[1] - 1,
                                        self.prey_loc[idx][1] + self.vision + 1)] != self.OUTSIDE_CLASS:
            self.prey_loc[idx][1] = min(self.dims[1] - 1,
                                        self.prey_loc[idx][1] + 1)

        # DOWN
        elif act == 2 and self.grid[min(self.dims[0] - 1,
                                        self.prey_loc[idx][0] + self.vision + 1),
                                    self.prey_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.prey_loc[idx][0] = min(self.dims[0] - 1,
                                        self.prey_loc[idx][0] + 1)

        # LEFT
        elif act == 3 and self.grid[self.prey_loc[idx][0] + self.vision,
                                    max(0,
                                        self.prey_loc[idx][1] + self.vision - 1)] != self.OUTSIDE_CLASS:
            self.prey_loc[idx][1] = max(0, self.prey_loc[idx][1] - 1)

    def _take_action(self, idx, act):
        # prey action
        if idx >= self.npredator:
            pass


        if self.reached_prey[idx] == 1:
            return

        # STAY action
        if act==4:
            return

        # UP
        if act==0 and self.grid[max(0,
                                self.predator_loc[idx][0] + self.vision - 1),
                                self.predator_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][0] = max(0, self.predator_loc[idx][0]-1)

        # RIGHT
        elif act==1 and self.grid[self.predator_loc[idx][0] + self.vision,
                                min(self.dims[1] -1,
                                    self.predator_loc[idx][1] + self.vision + 1)] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][1] = min(self.dims[1]-1,
                                            self.predator_loc[idx][1]+1)

        # DOWN
        elif act==2 and self.grid[min(self.dims[0]-1,
                                    self.predator_loc[idx][0] + self.vision + 1),
                                    self.predator_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][0] = min(self.dims[0]-1,
                                            self.predator_loc[idx][0]+1)

        # LEFT
        elif act==3 and self.grid[self.predator_loc[idx][0] + self.vision,
                                    max(0,
                                    self.predator_loc[idx][1] + self.vision - 1)] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][1] = max(0, self.predator_loc[idx][1]-1)

    def _get_reward(self):
        n = self.npredator if not self.enemy_comm else self.npredator + self.nprey
        reward = np.full(n, self.TIMESTEP_PENALTY)
        # print(np.all(self.predator_loc == self.prey_loc, axis=1))
        # print(self.predator_loc.shape, self.prey_loc.shape)
        on_prey = np.where(np.all(self.predator_loc == self.prey_loc, axis=1))[0]
        # prey에 도착한 predator의 숫자
        nb_predator_on_prey = on_prey.size

        if self.mode == 'cooperative':
            reward[on_prey] = self.POS_PREY_REWARD * nb_predator_on_prey
        elif self.mode == 'competitive':
            if nb_predator_on_prey:
                reward[on_prey] = self.POS_PREY_REWARD / nb_predator_on_prey
        elif self.mode == 'mixed':
            #reward[on_prey] = self.POS_PREY_REWARD * nb_predator_on_prey
            reward[on_prey] = self.PREY_REWARD
        else:
            raise RuntimeError("Incorrect mode, Available modes: [cooperative|competitive|mixed]")

        self.reached_prey[on_prey] = 1

        if np.all(self.reached_prey == 1) and self.mode == 'mixed':
            self.episode_over = True

        # Prey reward
        if nb_predator_on_prey == 0:
            reward[self.npredator:] = -1 * self.TIMESTEP_PENALTY
        else:
            # TODO: discuss & finalise
            reward[self.npredator:] = 0

        # Success ratio
        if self.mode != 'competitive':
            if nb_predator_on_prey == self.npredator:
                self.stat['success'] = 1
            else:
                self.stat['success'] = 0

        return reward

    def reward_terminal(self):
        return np.zeros_like(self._get_reward())


    def _onehot_initialization(self, a):
        ncols = self.vocab_size
        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[self._all_idx(a, axis=2)] = 1
        return out

    def _all_idx(self, idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    def render(self, mode='human', close=False):
        grid = np.zeros(self.BASE, dtype=object).reshape(self.dims)
        self.stdscr.clear()

        for p in self.predator_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'X'
            else:
                grid[p[0]][p[1]] = 'X'

        for p in self.prey_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'P'
            else:
                grid[p[0]][p[1]] = 'P'

        for row_num, row in enumerate(grid):
            for idx, item in enumerate(row):
                if item != 0:
                    if 'X' in item and 'P' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(3))
                    elif 'X' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(1))
                    else:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3),  curses.color_pair(2))
                else:
                    self.stdscr.addstr(row_num, idx * 4, '0'.center(3), curses.color_pair(4))

        self.stdscr.addstr(len(grid), 0, '\n')
        self.stdscr.refresh()

    def exit_render(self):
        curses.endwin()

    def get_env_info(self):
        return {"n_agents":self.npredator,
         "n_enemies": self.nprey,
         "node_features": 3,
         }