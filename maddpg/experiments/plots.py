
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sns.set(style="darkgrid")

# ## LOAD REWARDS
# rewards = []
# with open("./learning_curves/3a_3o_rewards_final.pkl", "rb") as f:
#     while True:
#         try:
#             rewards.append(pickle.load(f)[0])
#         except EOFError:
#             break

# LOAD AG REWARDS
ag1_agent0 = []
ag1_agent1 = []
ag1_agent2 = []
with open("./learning_curves/3a_0o_agrewards_final.pkl", "rb") as f:
    # print("==================== AG REWARDS =================== \n")
    while True:
        try:
            x0 = pickle.load(f)
            x1 = pickle.load(f)
            x2 = pickle.load(f)
            if x0:
                ag1_agent0.append(x0)
            if x1:
                ag1_agent1.append(x1)
            if x2:
                ag1_agent2.append(x2)
        except EOFError:
            break
# with open("./learning_curves/1a_0o_agrewards_final_x.pkl", "rb") as f:
#     # print("==================== AG REWARDS =================== \n")
#     while True:
#         try:
#             x0 = pickle.load(f)
#             x1 = pickle.load(f)
#             x2 = pickle.load(f)
#             if x0:
#                 ag1_agent0.append(x0)
#             if x1:
#                 ag1_agent1.append(x1)
#             if x2:
#                 ag1_agent2.append(x2)
#         except EOFError:
#             break
#
ag2_agent0 = []
ag2_agent1 = []
ag2_agent2 = []
with open("./learning_curves/3a_3o_agrewards_final.pkl", "rb") as f:
    # print("==================== AG REWARDS =================== \n")
    while True:
        try:
            x0 = pickle.load(f)
            x1 = pickle.load(f)
            x2 = pickle.load(f)
            if x0:
                ag2_agent0.append(x0)
            if x1:
                ag2_agent1.append(x1)
            if x2:
                ag2_agent2.append(x2)
        except EOFError:
            break

ag3_agent0 = []
ag3_agent1 = []
ag3_agent2 = []
with open("./learning_curves/3a_6o_agrewards_final.pkl", "rb") as f:
    # print("==================== AG REWARDS =================== \n")
    while True:
        try:
            x0 = pickle.load(f)
            x1 = pickle.load(f)
            x2 = pickle.load(f)
            if x0:
                ag3_agent0.append(x0)
            if x1:
                ag3_agent1.append(x1)
            if x2:
                ag3_agent2.append(x2)
        except EOFError:
            break

# ag6_agent0 = []
# ag6_agent1 = []
# ag6_agent2 = []
# with open("./learning_curves/6a_0o_agrewards_final.pkl", "rb") as f:
#     # print("==================== AG REWARDS =================== \n")
#     while True:
#         try:
#             x0 = pickle.load(f)
#             x1 = pickle.load(f)
#             x2 = pickle.load(f)
#             if x0:
#                 ag6_agent0.append(x0)
#             if x1:
#                 ag6_agent1.append(x1)
#             if x2:
#                 ag6_agent2.append(x2)
#         except EOFError:
#             break
# with open("./learning_curves/6a_0o_agrewards_final_x.pkl", "rb") as f:
#     # print("==================== AG REWARDS =================== \n")
#     while True:
#         try:
#             x0 = pickle.load(f)
#             x1 = pickle.load(f)
#             x2 = pickle.load(f)
#             if x0:
#                 ag6_agent0.append(x0)
#             if x1:
#                 ag6_agent1.append(x1)
#             if x2:
#                 ag6_agent2.append(x2)
#         except EOFError:
#             break


# ## LOAD ACTOR AND CRITIC LOSSES
# agent0_actor = []
# agent0_critic = []
# agent1_actor = []
# agent1_critic = []
# agent2_actor = []
# agent2_critic = []
# with open("./learning_curves/3a_3o_losses.pkl", "rb") as f:
#     # print("==================== LOSSES ===================\n")
#     while True:
#         try:
#             x0 = pickle.load(f)[0]
#             x1 = pickle.load(f)[0]
#             x2 = pickle.load(f)[0]
#             if x0:
#                 agent0_actor.append(x0[0])
#                 agent0_critic.append(x0[1])
#             if x1:
#                 agent1_actor.append(x1[0])
#                 agent1_critic.append(x1[1])
#             if x2:
#                 agent2_actor.append(x2[0])
#                 agent2_critic.append(x2[1])
#         except EOFError:
#             break


min_size = min(len(ag1_agent0), len(ag2_agent0),
               len(ag3_agent0))


# ### REWARD
# data = {"Reward": rewards, "Episodes": [i*1000 for i in range(len(rewards))]}
# plt.figure()
# plt.title("Reward")
# plt.xlabel("Episodes")
# plt.ylabel("Reward")
# sns.lineplot(data=data, y="Reward", x="Episodes")

# AGENTS REWARDS
ag1_reward = ag1_agent0[:min_size] + \
    ag1_agent1[:min_size] + ag1_agent2[:min_size]
data_ag1_rewards = {"AgReward": ag1_reward, "Episodes": [
    i*1000 for _ in range(3) for i in range(min_size)]}

ag2_reward = ag2_agent0[:min_size] + \
    ag2_agent1[:min_size] + ag2_agent2[:min_size]
data_ag2_rewards = {"AgReward": ag2_reward, "Episodes": [
    i*1000 for _ in range(3) for i in range(min_size)]}

ag3_reward = ag3_agent0[:min_size] + \
    ag3_agent1[:min_size] + ag3_agent2[:min_size]
data_ag3_rewards = {"AgReward": ag3_reward, "Episodes": [
    i*1000 for _ in range(3) for i in range(min_size)]}

# ag6_reward = ag6_agent0[:min_size] + \
#     ag6_agent1[:min_size] + ag6_agent2[:min_size]
# data_ag6_rewards = {"AgReward": ag6_reward, "Episodes": [
#     i*1000 for _ in range(3) for i in range(min_size)]}

plt.title("AgReward ")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.legend()
sns.lineplot(y="AgReward", x="Episodes",
             data=data_ag1_rewards, label="3 Ag & 0 Obs")
sns.lineplot(y="AgReward", x="Episodes",
             data=data_ag2_rewards, label="3 Ag & 3 Obs")
sns.lineplot(y="AgReward", x="Episodes",
             data=data_ag3_rewards, label="3 Ag & 6 Obs")
# sns.lineplot(y="AgReward", x="Episodes",
#              data=data_ag6_rewards, label="6 Ag & 0 Obs")

# ## ACTOR LOSSES
# losses = agent0_actor + agent1_actor + agent2_actor
# data_losses_actor = {"Losses": losses, "Episodes": [
#     i for _ in range(3) for i in range(len(agent0_actor))]}
# plt.figure()
# plt.title("ACtor Losses")
# plt.xlabel("Episodes")
# plt.ylabel("Actor Loss")
# sns.lineplot(y="Losses", x="Episodes", data=data_losses_actor)

# ## CRITIC LOSSES
# losses = agent0_critic + agent1_critic + agent2_critic
# data_losses_critic = {"Losses": losses, "Episodes": [
#     i for _ in range(3) for i in range(len(agent2_critic))]}
# plt.figure()
# plt.title("Critic Losses")
# plt.xlabel("Episodes")
# plt.ylabel("Critic Loss")
# sns.lineplot(y="Losses", x="Episodes", data=data_losses_critic)


plt.show()
