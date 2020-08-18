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

#BUFFER_SIZE = 5000000    #int(1e4)  # replay buffer size
#BATCH_SIZE = 512         #128        # minibatch size
#GAMMA = 0.987            # discount factor
#TAU = 0.001              # for soft update of target parameters
#LR_ACTOR = 0.0001        # learning rate of the actor 1e-4 
#LR_CRITIC = 0.0002       # learning rate of the critic 3e-4 
#WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'Multiagents for Colaboration'
class DDPGMultiAgent:
    def __init__(self, state_size,action_size,memory,num_agents,seed=1,p_gamma=0.997,p_tau=0.001,p_lrAct=0.0001,p_lrCritic=0.0001,mu=0.0,theta=0.17,sigma=0.24,istargetcopy=False,explore=0.3):
        super(DDPGMultiAgent,self).__init__()
      
        self.multiagent = [DDPGAgent (state_size,action_size,random_seed=seed,p_mu=mu,p_theta=theta, p_sigma=sigma,p_targetcopy=istargetcopy) for agent in range(num_agents)]
        print ("HP :Seed {},Gamma :{} , TAU:{}, LR_Act :{} , LR_Critic {} , Mu {}, Theta {}, Sigma {}, ExploreFactor {}, IsTargetHardcopy{}".format(seed,p_gamma,p_tau,p_lrAct,p_lrCritic,mu,theta,sigma,explore,istargetcopy))
        self.commonMemory =memory   # Replay memory

        #Hyper Params
        self.gamma = p_gamma
        self.random_seed =seed
        self.tau=p_tau
        self.mu = mu
        self.theta= theta
        self.sigma = sigma
        self.isHardcopy= istargetcopy
        self.explorfactor=explore

        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, self.random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, self.random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=p_lrAct)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, self.random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, self.random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=p_lrCritic, weight_decay=0.)

        # Noise process
        self.noise = OUNoise(action_size, self.random_seed, self.mu, self.theta, self.sigma)
        
        #Trained mode = true , copy local NN weights to target NN 
        if(self.isHardcopy):
            self.hard_update(self.actor_local,self.actor_target)
            self.hard_update(self.critic_local,self.critic_target)
    def __iter__(self):
        return [ agent for agent in self.multiagent]

    def act(self, env_states):
        actions = [ self.getActions(states) for agent,states in zip(self.multiagent,env_states)]
        return actions

     #Learn from Replay Memory
    def learn(self):
        #self.multiagent[agent_number].learn(experiences, self.gamma)
        #print("Check #2:Learning triggered thr DDPG Agent\n")
        [self.replayExp(self.commonMemory.sample(),self.gamma) for agent in self.multiagent]

    def resetNoise(self):
        [agent.resetNoise() for agent in self.multiagent]


    def getActions(self, states,add_noise=True):
        """Returns actions for given state as per current policy."""
          
        self.actor_local.eval()
        ##print("Multi Agent - act {}".format(states))
        states = torch.from_numpy(states).float().to(device)
        #states = torch.from_numpy(states).float().to(device)      
        ##state = torch.as_tensor(np.array(state).astype('float')).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            actions += self.explorfactor*self.noise.sample()   
        actions = np.clip(actions, -1, 1)
        
        return actions 


    def replayExp(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                     

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
    
    #https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L199
    def load_trained_model(self, actor_path, critic_path):
        print('Loading trained models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor_local.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic_local.load_state_dict(torch.load(critic_path))
    
    "When loading trained weights - copy ttarget netwrok also the same"
    def hard_update(self,local_model,target_model):
        for target_param,local_param in zip(target_model.parameters(),local_model.parameters()):
            target_param.data.copy_(local_param.data)
    