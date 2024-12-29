import gc
import os
import argparse
import random
import numpy as np
import torch
import gfootball.env as gf
from ac_modify import reward_func
from itertools import count
from ppo_agent import PPOAgent
from util import trans_img, plot_training


def set_seed(seed):
    # 设置 Python 随机种子
    random.seed(seed)

    # 设置 NumPy 随机种子
    np.random.seed(seed)

    # 设置 PyTorch 随机种子
    torch.manual_seed(seed)

    # 如果有 CUDA 设备，设置 PyTorch CUDA 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU

    # 保证每次运行时，CUDA 产生的随机数相同
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_env(args):
    env = gf.create_environment(
        env_name='11_vs_11_easy_stochastic',
        stacked=False,
        representation='raw',
        rewards='scoring',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=True,
        write_video=False,
        # logdir=os.path.join(logger.get_dir(), "model.pkl"),
        logdir='./',
        extra_players=None,
        number_of_left_players_agent_controls=3,
        number_of_right_players_agent_controls=0
    )
    # env = TransEnv(env)
    env.seed(args.seed)
    return env


def train(path, args):
    env = make_env(args)
    set_seed(args.seed)
    # single
    # num_actions = env.action_space.n
    # agent = PPOAgent(num_actions)

    # Get action space for each agent
    # multi
    num_actions = env.action_space.nvec  # MultiDiscrete action space
    num_agents = len(num_actions)
    # Create shared networks
    agent = PPOAgent(num_actions[0])  # Initialize with one action space

    reward_list = []

    if torch.cuda.is_available():
        print('cuda:', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name())
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    best_reward = float('-inf')  # Track the best reward
    best_model_paths = None  # To store paths of the best model

    for eps in range(args.episodes):
        # reset in each episode start
        obs = env.reset()
        env.seed(args.seed)
        # single
        # image = trans_img(obs[0]['frame'])
        # state = torch.FloatTensor([image])
        # Convert observations for each agent
        # multi
        states = [torch.FloatTensor([trans_img(obs[i]['frame'])]) for i in range(num_agents)]


        if torch.cuda.is_available():
            # single
            # state = state.cuda()
            # multi
            states = [s.cuda() for s in states]

        eps_reward = 0
        for t in count():
            # multi
            actions, action_probs = [], []
            # get next_state, reward

            for i in range(num_agents):
                action, action_prob = agent.select_action(states[i], obs[i])
                actions.append(action.item())
                action_probs.append(action_prob)

            # Execute actions in the environment
            next_obs, scores, done, _ = env.step(actions)

            next_states = [torch.FloatTensor([trans_img(next_obs[i]['frame'])]) for i in range(num_agents)]
            rewards = [reward_func(next_obs[i], scores[i], actions[i]) for i in range(num_agents)]
            eps_reward += sum(rewards)

            # single
            # action, action_prob = agent.select_action(state, next_obs)
            # next_obs, score, done, _ = env.step(action.item())
            # next_img = trans_img(next_obs[0]['frame'])
            # next_state = torch.FloatTensor([next_img])
            # reward = reward_func(next_obs, score, action.item())
            # eps_reward += reward

            # reward = torch.FloatTensor([reward])

            if torch.cuda.is_available():
                # single
                # action = action.cuda()
                # reward = reward.cuda()
                # next_state = next_state.cuda()
                # multi
                next_states = [s.cuda() for s in next_states]
                rewards = [torch.FloatTensor([r]).cuda() for r in rewards]

            # Store the transition in memory and update agent
            # ['state', 'action', 'log_prob', 'reward', 'next_state']
            # single
            # agent.memory_buffer.push(state, action, action_prob, reward, next_state)
            # multi
            for i in range(num_agents):
                agent.memory_buffer.push(
                    states[i], torch.LongTensor([actions[i]]).cuda(), action_probs[i], rewards[i], next_states[i]
                )

            # if reward > 0:
            #     print('in step{}: action:{}, reward:{}'.format(t, action, reward))
            # if sum(rewards) > 0:
            #     print('in step{}: actions:{}, rewards:{}'.format(t, actions, rewards))

            if done or t > 1500:  # Check if all agents are done
                break
            # single
            # state = next_state
            # multi
            states = next_states

        reward_list.append(eps_reward)
        agent.optimize_model(eps)
        # plot training rewards
        plot_training(reward_list, path)

        # # save model
        # if eps % 200 == 0 and eps > 0:
        #     actor_path = './checkpoints/' + str(eps) + '_actor_net.pkl'
        #     critic_path = './checkpoints/' + str(eps) + '_critic_net.pkl'
        #     # os.makedirs(model_path, exist_ok=True)
        #     torch.save(agent.actor_net.state_dict(), actor_path)
        #     torch.save(agent.critic_net.state_dict(), critic_path)

        # Check if this is the best reward and save the model
        if eps_reward > best_reward:
            best_reward = eps_reward
            best_model_paths = ('./best_actor_net.pkl', './best_critic_net.pkl')
            torch.save(agent.actor_net.state_dict(), best_model_paths[0])
            torch.save(agent.critic_net.state_dict(), best_model_paths[1])
            print("save best model")

        print('episode {}: last time {}, reward {}'.format(
            eps, t, eps_reward
        ))

    print('reward list', reward_list)
    print('Complete')
    env.close()


if __name__ == '__main__':
    # define parameter
    parser = argparse.ArgumentParser(description="PPO example")
    parser.add_argument('--seed', type=int, default=1024, metavar='seed')
    parser.add_argument('--learning_rate', type=float, default=0.01, metavar='lr')
    parser.add_argument('--gamma', type=float, default=0.8, metavar='gamma')
    parser.add_argument('--batch_size', type=int, default=128, metavar='batch')
    parser.add_argument('--episodes', type=int, default=200, metavar='episodes')
    parser.add_argument('--max_memory', type=int, default=10000, metavar='max memory')
    parser.add_argument('--ppo_update_time', type=int, default=10, metavar='ppo update')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, metavar='norm')
    parser.add_argument('--clip_param', type=float, default=0.2, metavar='clip')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')

    args = parser.parse_args()

    gc.collect()
    PATH = './PPO_plot/'
    os.makedirs(PATH, exist_ok=True)
    train(PATH, args)
