from agent import DDPGAgent
import torch
import numpy as np

class DDPGMultiAgent:
    def __init__(self, state_size,action_size,num_agents):
        super(DDPGMultiAgent,self).__init__()

        self.multiagent = [DDPGAgent (state_size,action_size) for agent in range(num_agents)]

    def act(self, env_states):
        actions = [ agent.act(states) for agent,states in zip(self.multiagent,env_states)]
        return actions

  
    def resetNoise(self):
        [agent.resetNoise() for agent in self.multiagent]