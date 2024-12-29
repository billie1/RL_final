import os
import argparse
import gfootball.env as gf
import torch
from ppo_agent import PPOAgent
from itertools import count
from util import trans_img, plot_training, save_data
from ac_modify import reward_func
from tqdm import tqdm


def make_env(seed):
    env = gf.create_environment(
        env_name='11_vs_11_easy_stochastic',
        stacked=False,
        representation='raw',
        rewards='scoring',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=True,
        write_video=False,
        logdir='./',
        extra_players=None,
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0
    )
    # env = TransEnv(env)
    env.seed(seed)
    return env



def test(path, args):
    # num_actions = 19
    # agent = PPOAgent(num_actions)
    # Get action space for each agent
    env = make_env(args.seed)
    num_actions = env.action_space.nvec  # MultiDiscrete action space
    num_agents = len(num_actions)
    # Create shared networks
    agent = PPOAgent(num_actions[0])  # Initialize with one action space
    agent.test_model(args.saved_path)


    if torch.cuda.is_available():
        print('cuda:', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name())
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    reward_list = []
    for match in tqdm(range(10), desc="Matches"):
        obs = env.reset()
        env.seed(args.seed)
        # image = trans_img(obs[0]['frame'])
        # state = torch.FloatTensor([image])
        # Convert observations for each agent
        states = [torch.FloatTensor([trans_img(obs[i]['frame'])]) for i in range(num_agents)]
        next_obs = obs

        if torch.cuda.is_available():
            # state = state.cuda()
            states = [s.cuda() for s in states]

        eps_reward = 0
        for t in count():
            actions, action_probs = [], []
            # get next_state, reward
            # action, action_prob = agent.select_action(state, next_obs)
            # next_obs, score, done, _ = env.step(action.item())
            # next_img = trans_img(next_obs[0]['frame'])
            # next_state = torch.FloatTensor([next_img])
            # reward = reward_func(next_obs, score, action.item())
            # eps_reward += reward
            for i in range(num_agents):
                action, action_prob = agent.select_action(states[i], obs[i])
                actions.append(action.item())
                action_probs.append(action_prob)

            # Execute actions in the environment
            next_obs, scores, done, _ = env.step(actions)

            next_states = [torch.FloatTensor([trans_img(next_obs[i]['frame'])]) for i in range(num_agents)]
            rewards = [reward_func(next_obs[i], scores[i], actions[i]) for i in range(num_agents)]

            eps_reward += sum(rewards)
            if torch.cuda.is_available():
                # next_state = next_state.cuda()
                next_states = [s.cuda() for s in next_states]

            if done or t > 1500:  # Check if all agents are done
                break
            states = next_states
            # state = next_state
        print(match, ": ", eps_reward)
        reward_list.append(eps_reward)

        # plot training rewards
        plot_training(reward_list, path)

    # save reward list
    save_data(reward_list, 'test_score', 'test_score.xlsx')

    print('Complete')
    env.close()


if __name__ == '__main__':
    # define parameter
    parser = argparse.ArgumentParser(description="PPO test")
    parser.add_argument('--seed', type=int, default=1024, metavar='seed')
    parser.add_argument('--saved_path', type=str, default='./best', metavar='path')

    args = parser.parse_args()

    PATH = 'PPO_plot_test/'
    os.makedirs(PATH, exist_ok=True)
    test(PATH, args)
