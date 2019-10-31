#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Christian Heider Nielsen"

from matplotlib import pyplot
import numpy


def plot_figure(episodes, eval_rewards, env_id):
    episodes = numpy.array(episodes)
    eval_rewards = numpy.array(eval_rewards)


numpy.savetxt(f"./output/{env_id}_ppo_episodes.txt", episodes)
numpy.savetxt(f"./output/{env_id}_ppo_eval_rewards.txt", eval_rewards)

pyplot.figure()
pyplot.plot(episodes, eval_rewards)
pyplot.title("%s" % env_id)
pyplot.xlabel("Episode")
pyplot.ylabel("Average Reward")
pyplot.legend(["PPO"])
pyplot.savefig(f"./output/{env_id}_ppo.png")
