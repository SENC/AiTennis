# AiTennis

# AI - Deep Reinforcement Learning (DRL) - Continuous space - Policy based - 'Actor-Critic' - DDPG :
Deep Reinforcement Learning (DRL) ,In simple , a mathematical way to clone the experiences start with trial and error (reward & punishment) approach to form a #digitalmemory (Policy with Critic) and take the best actions based on current situation to maximize the rewards like how a Toddler learn to play a Cricket  motivated by 'high scorer' achievement.

This project simply help you to get the core of how AI works and detail implementation of Deep Deterministic Policy Gradients (DDPG) algorithm .


# 9 steps for AI to Win n Continuous...

<img src=images/9StepsDDPG_ActorCriticv2.png width="684">

# AITennsi Project Overview

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.  The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.


The Environment preview

<img src=images/Reacher.gif width="684">

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

# Problem Statement
Develop an colloaborative AI Agents using 'actor-critic' methods - which should learn the best policy to maximize its rewards by taking best actions in the given continuous environment  


# Files :

1. Report.pdf: Gives complete project implementation report with results
2. DDPG/Continuous_Control.ipynb : Python Notebook "DDPG Implementation  in windows env"
3. DDPG/agent.py : DDPG Agent class defintion and Utility functions like Replay Memory and OUNoise funtions
4. DDPG/nn_model.py : Actor and Critic Neural Network Architecture
5. DDPG/checkpoint_actor.pth : Trained Agent's Neural Network weights - Actor (Ploicy)
6. DDPG/checkpoint_critic.pth: Trained Agent's Neural Network weights - Critic (state-action Value)
7. DDPG/Tennis.exe : Unity Environment for Windows- 64bit



# Environment Setup:
For Environment setup, you need Python 3.6, pytorch ,Ml-agents, Unity Environment to be installed. Once you setup your python environment , you just needs to have agent.py ,nn_model.py and Continuous_Control.ipynb if you wish to have quick hands-on .
1. Anaconda Navigator - python 3.6
2. PyTorch 
 Example: conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
 select right commands based on your OS , Python and conda/pip 
  https://pytorch.org/get-started/locally/>
3. Unity Environment 
   https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md
   
4.More detail on  Ml-Agents 
   https://github.com/Unity-Technologies/ml-agents/blob/master/docs/ML-Agents-Overview.md

5.Windows Env setup details - deprecated but may be useful
  https://hub.udacity.com/rooms/community:nd893:845401-project-503-smg-2/community:thread-12988088548-3621024?messageId=3652300&contextType=room

  
For the detail instruction please check https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md

