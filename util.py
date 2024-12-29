import gym
import cv2
import numpy as np
import pandas as pd
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt


# build replay buffer
class ReplayBuffer(object):
    def __init__(self, capacity):
        # define the max capacity of memory
        self.memory = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition',
                                     ['state', 'action', 'log_prob', 'reward', 'next_state'])

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        # print(self.Transition(*args))
        self.memory.append(self.Transition(*args))

    def sample(self, bach_size):
        return random.sample(self.memory, bach_size)

    def clear(self):
        self.memory.clear()


# resize and transpose image
def trans_img(image):
    height = 72
    width = 128
    image = cv2.resize(image,
                       (width, height),
                       interpolation=cv2.INTER_AREA
                       )
    image = image.transpose(2, 0, 1)
    image = np.ascontiguousarray(image, dtype=np.float32) / 255
    return image


# resize and transpose observation
class TransEnv(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.height = 72
        self.width = 96
        self.channel = 4
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.channel, self.height, self.width),
            dtype=np.uint8
        )

    def observation(self, observation):
        obs = cv2.resize(observation,
                         (self.width, self.height),
                         interpolation=cv2.INTER_AREA
                         )
        return obs.reshape(self.observation_space.low.shape)


# plot rewards
def plot_training(rewards, path):
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title('Training')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run Time')
    ax.plot(rewards)
    run_time = len(rewards)

    # path = './AC_CartPole-v0/' + str(RunTime) + '.jpg'
    path = path + str(run_time) + '.jpg'
    if run_time % 100 == 0:
        plt.savefig(path)
    plt.close()
    # plt.pause(0.000001)


# save to local excel
def save_data(data, column_name, path):
    df = pd.DataFrame(data, columns=[column_name])
    df.to_excel(path, index=False)


action_count = 0
last_action = 0


# get raw observations
class GetRawObservations:
    def __init__(self, obs):
        self.ball_owned_team = obs[0]['ball_owned_team']
        self.ball_position = obs[0]['ball']
        self.ball_owned_player = obs[0]['ball_owned_player']
        self.ball_direction = obs[0]['ball_direction']
        self.left_team_direction = obs[0]['left_team_direction']
        self.left_team_position = obs[0]['left_team']
        self.right_team_position = obs[0]['right_team']
        self.designated_player = obs[0]['designated']

    def get_ball_info(self):
        return self.ball_owned_team, self.ball_position, self.ball_direction, self.ball_owned_player

    def get_team_position(self):
        return self.left_team_position, self.right_team_position

    def get_team_direction(self):
        return self.left_team_direction

    def get_player(self):
        return self.designated_player
    
    
class GetMutliRawObservations:
    def __init__(self, obs):
        self.ball_owned_team = obs['ball_owned_team']
        self.ball_position = obs['ball']
        self.ball_owned_player = obs['ball_owned_player']
        self.ball_direction = obs['ball_direction']
        self.left_team_direction = obs['left_team_direction']
        self.left_team_position = obs['left_team']
        self.right_team_position = obs['right_team']
        self.designated_player = obs['designated']

    def get_ball_info(self):
        return self.ball_owned_team, self.ball_position, self.ball_direction, self.ball_owned_player

    def get_team_position(self):
        return self.left_team_position, self.right_team_position

    def get_team_direction(self):
        return self.left_team_direction

    def get_player(self):
        return self.designated_player


# count repeated action
def collect_action(action):
    global action_count
    global last_action

    if action == last_action:
        action_count += 1

    # if repeat over 10 step
    # modify will trigger to prevent fall into local minimum
    if action_count >= 20:
        action_count = 0
        last_action = action
        return action, True
    else:
        last_action = action
        return action, False
