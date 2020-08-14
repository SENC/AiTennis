from agent import DDPGAgent
import torch
import numpy as np

class DDPGMultiAgent:
    def __init__(self, state_size,action_size,num_agents):
        super(DDPGMultiAgent,self).__init__()
        self.gamma = 0.997
        self.multiagent = [DDPGAgent (state_size,action_size) for agent in range(num_agents)]
        
    def __iter__(self):
        return [ agent for agent in self.multiagent]

    def act(self, env_states):
        actions = [ agent.act(states) for agent,states in zip(self.multiagent,env_states)]
        return actions

     #Learn from Replay Memory
    def learn(self, experiences,gamma):
        #self.multiagent[agent_number].learn(experiences, self.gamma)
        [agent.learn(experiences,gamma) for agent in self.multiagent]

    def resetNoise(self):
        [agent.resetNoise() for agent in self.multiagent]