from agent import DDPGAgent
import torch
import numpy as np

class DDPGMultiAgent:
    def __init__(self, state_size,action_size,memory,p_gamma,num_agents):
        super(DDPGMultiAgent,self).__init__()
      
        self.multiagent = [DDPGAgent (state_size,action_size) for agent in range(num_agents)]
        self.gamma = p_gamma
        self.commonMemory =memory   # Replay memory
        
        
    def __iter__(self):
        return [ agent for agent in self.multiagent]

    def act(self, env_states):
        actions = [ agent.act(states) for agent,states in zip(self.multiagent,env_states)]
        return actions

     #Learn from Replay Memory
    def learn(self):
        #self.multiagent[agent_number].learn(experiences, self.gamma)
        #print("Check #2:Learning triggered thr DDPG Agent\n")
        [agent.learn(self.commonMemory.sample(),self.gamma) for agent in self.multiagent]

    def resetNoise(self):
        [agent.resetNoise() for agent in self.multiagent]