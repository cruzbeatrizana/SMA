#!/usr/bin/env python

import os
import argparse
from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy, Navigation
import multiagent.scenarios as scenarios
import sys
import time
import numpy as np
import math
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


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py',
                        help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation, info_callback=None, shared_viewer=True)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    # policies = [InteractivePolicy(env, i) for i in range(env.n)]
    policies = [Navigation(env) for _ in range(env.n)]
    # execution loop
    Q = np.zeros((9999, 5))

    learning_rate = 0.5
    gamma = 0.5
    iterations = 1000
    while True:
        n = 0
        obs_n = env.reset()
        while n < iterations:
            # query for action from each agent's policy
            states = get_state(obs_n)  # State i
            act_n = actions(Q, states)  # Actions i

            act_u = []
            for i, act in enumerate(act_n):
                u = np.zeros(5)
                u[act] = 1
                act_u.append(np.concatenate([u, np.zeros(env.world.dim_c)]))

            # step environment
            obs_n, reward_n, done_n, _ = env.step(act_u)
            states_ = get_state(obs_n)  # State i+1
            act_n_ = actions(Q, states)  # Actions i+1
            # print(reward_n)

            for i, state in enumerate(states):
                Q[state, act_n[i]] = Q[state, act_n[i]] + learning_rate * \
                    (reward_n[i] + gamma * Q[states_[i],
                                             act_n_[i]] - Q[state, act_n[i]])

            # render all agent views
            env.render()
            # display rewards
            # for agent in env.world.agents:
            #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
            n = n + 1
