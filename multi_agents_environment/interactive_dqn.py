#!/usr/bin/env python

import os
import argparse
from multiagent.environment import MultiAgentEnv
from multiagent.A2C import A2C
from multiagent.policy import InteractivePolicy, Navigation
import multiagent.scenarios as scenarios
import sys
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf

import os
root_logdir = os.path.join(os.curdir, "my_logs")


def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()  # e.g., './my_logs/run_2019_06_07-15_15_22'


print("Python version")
print(sys.version)


sys.path.insert(1, os.path.join(sys.path[0], '..'))


def get_state(n_obs):
    states = []
    for i, obs in enumerate(n_obs):
        distance = obs[4 + 2*i: 4 + 2*i + 2]
        states.append(
            min(int(math.sqrt(distance[0] ** 2 + distance[1] ** 2)*1000), 9998))

    return states


def actions(Q, states):
    n_act = []
    eps = 0.9
    for state in states:
        expected_rewards = Q[state]  # Q(S) = [E0, E1, E2....]
        # A = argmax(Q(S))
        n_act.append(np.random.choice(range(5)) if np.random.random()
                     < eps else np.argmax(expected_rewards))

    return n_act


# if __name__ == '__main__':
#     # parse arguments
#     parser = argparse.ArgumentParser(description=None)
#     parser.add_argument('-s', '--scenario', default='simple.py',
#                         help='Path of the scenario Python script.')
#     args = parser.parse_args()

#     # load scenario from script
#     scenario = scenarios.load(args.scenario).Scenario()
#     # create world
#     world = scenario.make_world(num_agents=2)
#     # create multiagent environment
#     env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
#                         scenario.observation, info_callback=None, done_callback=scenario.done, shared_viewer=True)
#     # render call to create viewer window (necessary only for interactive policies)
#     env.render()
#     # create interactive policies for each agent
#     # policies = [InteractivePolicy(env, i) for i in range(env.n)]
#     policies = [Navigation(env, import_model=True, num=i)
#                 for i in range(env.n)]
#     # execution loop

#     n = 0
#     done_n = [False]
#     obs_n = env.reset()
#     while n < 200 or (True in done_n):
#         env.render()
#         # query for action from each agent's policy
#         act_u = [policy.action(obs_n[i]) for i, policy in enumerate(policies)]

#         # step environment
#         obs_n_, reward_n_, done_n, _ = env.step(act_u)
#         for agent, obs, action, reward, obs_, done in zip(policies, obs_n, act_u, reward_n_, obs_n_, done_n):
#             action = np.argmax(action)
#             agent.store_experience(obs, action, reward, obs_, done)

#             # if n % 64 == 0:
#             #    agent.train()

#         # print(history)
#         n = n + 1
#         obs_n = obs_n_
#         # if n == 5:

#     exit(-1)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py',
                        help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world(num_agents=3)
    # create multiagent environment
    env = MultiAgentEnv(world, reset_callback=scenario.reset_world, reward_callback=scenario.reward,
                        observation_callback=scenario.observation, info_callback=None, done_callback=scenario.done, shared_viewer=True, width=1000, height=1000)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    # policies = [InteractivePolicy(env, i) for i in range(env.n)]
    # policy = Navigation(env, lr=1e-4, batch_size=256, run_logdir=run_logdir)

    state_dim = np.size(env.observation_space)
    action_dim = np.size(env.action_space)

    a2c = A2C(env, logdir=run_logdir)

    # execution loop

    learning_rate = 0.5
    gamma = 0.5
    g = 0
    iterations = 500
    max_games = 10000
    history = []
    cum_reward_1 = 0
    cum_reward_2 = 0
    cum_reward_3 = 0
    done_n = [False]
    plt.figure("Rewards")
    checkpoint_path = "./my_dqn.ckpt"

    a2c.train()

    # while g < max_games:
    #     n = 1
    #     obs_n = env.reset()
    #     while n < iterations or (True in done_n):
    #         env.render()
    #         # query for action from each agent's policy
    #         act_u = [policy.action(obs) for obs in obs_n]
#
    #         # step environment
    #         obs_n_, reward_n_, done_n, _ = env.step(act_u)
    #         for obs, action, reward, obs_, done in zip(obs_n, act_u, reward_n_, obs_n_, done_n):
    #             action = np.argmax(action)
    #             policy.store_experience(obs, action, reward, obs_, done)
#
    #             if n % 256 == 0:
    #                 policy.train()
#
    #         # print(history)
    #         cum_reward_1 += reward_n_[0]
    #         n = n + 1
    #         obs_n = obs_n_
    #         # print(reward_n_)
    #         # exit(-1)
#
    #     g += 1
#
    #     with policy.train_summary_writer.as_default():
    #         tf.summary.scalar('cumulative_reward', cum_reward_1, step=g)
#
    #     history.append(cum_reward_1)
    #     cum_reward_1 = 0
#
    # policy.actor_network.save("actor_agent.h5")
    # policy.critic_network.save("critic_agent.h5")
#
    # plt.plot(history, label="Agent " + str(i))
    # plt.legend(["Agent " + str(i) for i, _ in enumerate(history)])
    # plt.show()
