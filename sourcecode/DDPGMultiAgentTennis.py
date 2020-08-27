import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import DDPGAgent
from agent import OUNoise
from nn_model import Actor, Critic


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'Multiagents for Colaboration'
class DDPGMultiAgent:
    def __init__(self, state_size,action_size,memory,num_agents,config):
        super(DDPGMultiAgent,self).__init__()
      
        self.commonMemory =memory   # Replay memory

        #Hyper Params
        
        self.random_seed = config["SEED"]
        self.gamma = config["GAMMA"]
        self.tau=config["TAU"]
        self.lrActor = config["LR_ACTOR"]
        self.lrCritic =config["LR_CRITIC"]
        
        self.mu = config["MU"]
        self.theta= config["THETA"]
        self.sigma = config["SIGMA"]
        self.isNNHardcopy= False
        self.explorfactor=config["EXPLORE"]
        self.micro_batch_size= config["BATCH_SIZE"]

        self.num_agents=num_agents
        self.action_size=action_size
        self.state_size=state_size

        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, self.random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, self.random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lrActor)

        # Critic Network (w/ Target Network)
        #self.critic_local = Critic(state_size, action_size, self.random_seed).to(device)
        #self.critic_target = Critic(state_size, action_size, self.random_seed).to(device)
        #self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lrCritic, weight_decay=0.)

        #self.multiagent = [DDPGAgent (state_size,action_size,self.actor_local, self.actor_target,self.actor_optimizer,self.critic_local,self.critic_target,self.critic_optimizer,p_gamma=self.gamma,p_lr_actor= self.lrActor,p_lr_critic= self.lrCritic,p_seed=self.p_seed,p_mu=self.mu,p_theta=self.theta, p_sigma=self.sigma,p_tau=self.tau ,p_targetcopy=self.isNNHardcopy,p_explore=self.explorfactor) for agent in range(num_agents)]
        #Actor is Common for Both Agents and Target for each Agent
        self.multiagent = [DDPGAgent (state_size,action_size,self.actor_local, self.actor_target,self.actor_optimizer,config) for agent in range(num_agents)]
         
        # Noise process
        self.noise = OUNoise(action_size, self.random_seed, self.mu, self.theta, self.sigma)
        
        #Trained mode = true , copy local NN weights to target NN 
        if(self.isNNHardcopy):
            self.hard_update(self.actor_local,self.actor_target)
            self.hard_update(self.critic_local,self.critic_target)
        print ("HP :Gamma :{} , TAU:{}, LR_Act :{} , LR_Critic {} , Mu {}, Theta {}, Sigma {}, ExploreFactor {}, IsTargetHardcopy{}".format(self.gamma,self.tau,self.lrActor,self.lrCritic,self.mu,self.theta,self.sigma,self.explorfactor,self.isNNHardcopy))
        
    def __iter__(self):
        return [ agent for agent in self.multiagent]

    
    'Take action for each agents in the MultiAgent instance'
    def act(self, env_states):
        actions = np.zeros([self.num_agents,self.action_size])
        #[agent.act(env_states[i] for i, agent in self.multiagent)]
        
        for i, agent in enumerate(self.multiagent):
            actions[i, :]=agent.act(env_states[i])
        #Not working since states for each agent seems some bug while unwinds
        #actions = [ self.getActions(agent,states) for agent,states in zip(self.multiagent,env_states)]
        #Define array for each agent(s) with its actions_dimension
       
        return actions

    
    'Learn from Replay Memory'
    def learn(self):
        
        #print("Check #2:Learning triggered thr DDPG Agent\n")
        #[agent.learn(self.commonMemory.sample(),self.gamma,self.tau) for agent in self.multiagent]
        [agent.learn(self.commonMemory.sample(),agent.gamma,agent.tau) for agent in self.multiagent]
        
    'Recollect the experiences & update target netowrk from common memory '
    def replayExp(self,agent, experiences, gamma):
        
        states, actions, rewards, next_states, dones = experiences
        
        # Get predicted next-state actions and Q values from target models
        actions_next = agent.actor_target(next_states)
        Q_targets_next = agent.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = agent.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = agent.actor_local(states)
        actor_loss = -agent.critic_local(states, actions_pred).mean()
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        agent.soft_update(agent.critic_local, agent.critic_target, self.tau)
        agent.soft_update(agent.actor_local, agent.actor_target, self.tau) 
    
        
    'Reset Agents noise'
    def resetNoise(self):
        [agent.reset() for agent in self.multiagent]

    'Common Actor Network for both agents'
    def getActions_v1(self, states,add_noise=True):
        """Returns actions for given state as per current policy."""
          
        #self.actor_local.eval()
        ###print("Multi Agent - act {}".format(states))
        states = torch.from_numpy(states).float().to(device)
        #states = torch.from_numpy(states).float().to(device)      
        ##state = torch.as_tensor(np.array(state).astype('float')).to(device)
        #self.actor_local.eval()
        #with torch.no_grad():
            #actions = self.actor_local(states).cpu().data.numpy()
        
        #self.actor_local.train()
        actions = self.actor_local(states).cpu().data.numpy()
        if add_noise:
            actions += self.explorfactor*self.noise.sample()   
        actions = np.clip(actions, -1, 1)
        
        return actions 

    'Common Actor Network for both agents'
    def getActions(self,agent, states,add_noise=True):
        """Returns actions for given state as per current policy."""

        agent.actor_local.eval()  
        states = torch.from_numpy(states).float().to(device)
        with torch.no_grad():
            actions = agent.actor_local(states).cpu().data.numpy()
        agent.actor_local.train()
        if add_noise:
            actions += self.explorfactor*self.noise.sample()   
        actions = np.clip(actions, -1, 1)
        
        return actions 
    
    'Soft update from local to target network'
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    'Load trained /saved weighted NN model'
    def load_trained_model(self, actor_path, critic_path,actor_tar_path,critic_tar_path):
        print('Loading trained models from {} and {}'.format(actor_path, critic_path,actor_tar_path,critic_tar_path))
        for agent in self.multiagent:
            if actor_path is not None:
                agent.actor_local.load_state_dict(torch.load(actor_path))
            if critic_path is not None: 
                agent.critic_local.load_state_dict(torch.load(critic_path))
            if actor_tar_path is not None:
                agent.actor_target.load_state_dict(torch.load(actor_tar_path))
            if critic_tar_path is not None: 
                agent.critic_target.load_state_dict(torch.load(critic_tar_path))
        print('Done!')    
    def load_trained_model_local(self, actor_path, critic_path):
        print('Loading trained models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor_local.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic_local.load_state_dict(torch.load(critic_path))
        

    "When loading trained weights - copy ttarget netwrok also the same"
    def hard_update(self,local_model,target_model):
        for target_param,local_param in zip(target_model.parameters(),local_model.parameters()):
            target_param.data.copy_(local_param.data)

    "Below methods for MultiAgent has individual agents and load their corresponding weights"
    def load_trained_model_v1(self, actor_path, critic_path):
        [ agent.load_trained_model(actor_path,critic_path) for agent in self.multiagent]

    "Get the Actor Network properties of multiagent "
    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor_local for ddpg_agent in self.multiagent]
        return actors
    
    'Get the Critic Network property of multiagent'
    def get_critics(self):
        """get critics of all the agents in the MADDPG object"""
        critics = [ddpg_agent.critic_local for ddpg_agent in self.multiagent]
        return critics

    


            
    
    
    