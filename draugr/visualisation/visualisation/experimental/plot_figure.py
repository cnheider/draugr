#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Christian Heider Nielsen"

import matplotlib.pyplot as plt
import numpy


def plot_figure(episodes, eval_rewards, env_id):
    episodes = numpy.array(episodes)
    eval_rewards = numpy.array(eval_rewards)


numpy.savetxt(f"./output/{env_id}_ppo_episodes.txt", episodes)
numpy.savetxt(f"./output/{env_id}_ppo_eval_rewards.txt", eval_rewards)

plt.figure()
plt.plot(episodes, eval_rewards)
plt.title("%s" % env_id)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.legend(["PPO"])
plt.savefig(f"./output/{env_id}_ppo.png")
