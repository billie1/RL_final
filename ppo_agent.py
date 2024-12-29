import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
from util import ReplayBuffer
from nn_layer import PPOActor, PPOCritic


class PPOAgent:
    def __init__(
            self,
            num_actions=19,
            max_memory=10000,
            batch_size=64,
            gamma=0.9,
            ppo=10,
            grad_norm=0.5,
            clip=0.2
    ):
        super(PPOAgent, self).__init__()

        self.ActionTransition = namedtuple(
            'Transition',
            ['state', 'action', 'log_prob', 'reward', 'next_state']
        )

        # self.ActionTuple = namedtuple('Action', ['log_prob', 'value'])

        self.actor_net = PPOActor(num_actions)
        self.critic_net = PPOCritic()
        if torch.cuda.is_available():
            self.actor_net.cuda()
            self.critic_net.cuda()

        self.actor_optimizer = optim.Adam(self.actor_net.parameters())
        self.critic_optimizer = optim.Adam(self.critic_net.parameters())
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.ppo_update_time = ppo
        self.max_grad_norm = grad_norm
        self.clip_param = clip
        self.memory_buffer = ReplayBuffer(max_memory)
        self.writer = SummaryWriter('./tensorboardX')
        self.training_step = 0

    def select_action(self, state, obs):
        with torch.no_grad():
            prob = self.actor_net(state)

        c = Categorical(prob)
        sample_action = c.sample()
        # action_prob = prob[:, sample_action.item()].item()
        action_prob = prob[:, sample_action.item()]


        if torch.cuda.is_available():
            sample_action = sample_action.cuda()
            action_prob = action_prob.cuda()


        return sample_action, action_prob

    def get_value(self, state):
        with torch.no_grad():
            state_value = self.critic_net(state)

        if torch.cuda.is_available():
            state_value = state_value.cuda()

        return state_value

    def loss_function(self, eps):
        # get train batch (state, action, reward) from replay buffer
        trans_tuple = self.memory_buffer.sample(self.batch_size)
        batch = self.ActionTransition(*zip(*trans_tuple))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        # value_batch = torch.cat(batch.value)
        reward_batch = torch.cat(batch.reward)
        log_prob_batch = torch.cat(batch.log_prob)

        # Discounted rewards and Advantage Calculation
        R = 0
        total_rewards = []
        eps = np.finfo(np.float32).eps.item()

        # Compute discounted rewards in reverse order (from t to 0)
        reward_batch_reversed = reward_batch.flip(0)
        reward_batch_reversed.tolist()
        for r in reward_batch_reversed:
            R = r + self.gamma * R
            total_rewards.insert(0, R)
        # for r in reward_batch[::-1]:
        #
        #     R = r + self.gamma * R
        #     total_rewards.insert(0, R)

        total_rewards = torch.tensor(total_rewards)
        # Normalize total_rewards for better stability
        total_rewards = (total_rewards - total_rewards.mean()) / (total_rewards.std() + eps)

        for i in range(self.ppo_update_time):
            # Sample a batch for training
            for index in BatchSampler(SubsetRandomSampler(range(len(batch))), self.batch_size, False):

                if self.training_step % 1000 == 0:
                    print('I_ep {} ，train {} times'.format(eps, self.training_step))

                # Calculate advantage: U = total_rewards, V = critic(state)
                U = total_rewards[index].view(-1, 1).cuda()  # Target (Discounted rewards)
                V = self.critic_net(state_batch[index]).cuda()  # Critic's value estimation
                advantage = U - V  # Advantage function

                # PPO loss calculation
                # action_prob = self.actor_net(state_batch[index]).gather(1, action_batch[index])  # new policy
                action_prob = self.actor_net(state_batch[index]).gather(1,
                                                                        action_batch[index].unsqueeze(1))  # new policy

                ratio = (action_prob / log_prob_batch[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # Policy loss (maximize clipped objective)
                policy_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN descent
                self.writer.add_scalar('loss/action_loss', policy_loss, global_step=self.training_step)

                # Value loss (mean squared error between the target and the estimated value)
                value_loss = func.mse_loss(U, V)
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)

                self.training_step += 1

        return policy_loss, value_loss

    def optimize_model(self, eps):

        # collect enough experience data
        if len(self.memory_buffer) < self.batch_size:
            return

        policy_loss, value_loss = self.loss_function(eps)

        # update actor network
        self.actor_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # update critic network
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # Clear experience
        self.memory_buffer.clear()

    def test_model(self, saved_path):
        if torch.cuda.is_available():
            self.actor_net.cuda()
            self.actor_net.load_state_dict(torch.load("{}_actor_net.pkl".format(saved_path)))
        else:
            self.actor_net.load_state_dict(torch.load("{}/2000_actor_net.pkl".format(saved_path)))

        self.actor_net.eval()
