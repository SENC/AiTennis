import numpy as np
import random
import copy
from collections import namedtuple, deque

from nn_model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = 5000000    #int(1e4)  # replay buffer size
BATCH_SIZE = 1024         #128        # minibatch size
GAMMA = 0.996            # discount factor
TAU = 0.001              # for soft update of target parameters
LR_ACTOR = 0.0002        # learning rate of the actor 1e-4 
LR_CRITIC = 0.0001       # learning rate of the critic 3e-4 
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size,nn_actor_local,nn_actor_target,nn_actor_opt,config):
    
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(DDPGAgent,self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = config["SEED"]

        #Hyper Parameters
        self.seed = config["SEED"]
        self.gamma = config["GAMMA"]
        self.lrActor = config["LR_ACTOR"]
        self.lrCritic =config["LR_CRITIC"]
        self.mu = config["MU"]
        self.theta= config["THETA"]
        self.sigma = config["SIGMA"]
        
        self.tau=config["TAU"]
        
        self.explorfactor=config["EXPLORE"]
        self.ishardcopy = True        
        #Actot Network - Common for Multiple agent
        self.actor_local = nn_actor_local
        self.actor_target =nn_actor_target
        self.actor_optimizer = nn_actor_opt

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.state_size, self.action_size, self.random_seed).to(device) #nn_critic_local
        self.critic_target = Critic(self.state_size, self.action_size,self.random_seed).to(device)#nn_critic_target
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lrCritic, weight_decay=WEIGHT_DECAY) #nn_critic_opt

        

       
        # Noise process
        self.noise = OUNoise(action_size,self.seed,mu=self.mu,theta=self.theta,sigma=self.sigma)

        #Trained mode = true , copy local NN weights to target NN 
        if(self.ishardcopy):
            self.hard_update(self.actor_local,self.actor_target)
            self.hard_update(self.critic_local,self.critic_target)
        
        #Print the HP Value for confirmation
        print ("Agent's HP :Seed {},Gamma :{}, LR_Act :{} , LR_Critic {} , Mu {}, Theta {}, Sigma {},TAU:{}, ExploreFactor {}, IsTargetHardcopy{}".format(self.seed,self.gamma,self.lrActor,self.lrCritic,self.mu,self.theta,self.sigma,self.tau, self.explorfactor,self.ishardcopy))
       

    'Agent Act -Policy based actions '
    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
          
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy() 
        self.actor_local.train()
        if add_noise:
            actions += self.explorfactor*self.noise.sample()         
        actions = np.clip(actions, -1, 1)
        
        return actions

    
    def step(self,common_memory,gamma=0.99,tau=0.0013):
        
        experiences = common_memory.sample()
        self.learn(experiences,gamma,tau)
        
    def target_act(self, states, noise=0.0):
        states = torch.from_numpy(states).float().to(device)
        actions = self.actor_target(states) + self.explorfactor*self.noise.sample()
        return actions

    def reset(self):
        self.noise.reset()

    "DEEP Q Learning "
    def learn(self, experiences, gamma,tau):
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

        #States contains 2 agents observations from the env
        #Concatenates the given sequence of state tensors in to 1 dimension
        #tensor([[agent1State],agent2stats]])
        states_tensor = torch.cat(states,dim=1).to(device)
        actions_tensor = torch.cat(actions,dim=1).to(device)
        next_states_tensor = torch.cat(next_states,dim=1).to(device)

        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        #print("Next states",next_states)
        actions_next_local = [self.actor_target(next_state) for next_state in next_states]
        actions_next_tensor = torch.cat(actions_next_local,dim=1).to(device)
        #print(actions_next)
        
        # Compute Q targets for current states (y_i)
        Q_targets_next = self.critic_target.forward(next_states_tensor,actions_next_tensor)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local.forward(states_tensor, actions_tensor)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = [self.actor_local(state) for state in states]
        actions_pred_tensor =torch.cat(actions_pred,dim=1).to(device)
        actor_loss = -self.critic_local(states_tensor, actions_pred_tensor).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(),0.5)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, tau)
        self.soft_update(self.actor_local, self.actor_target, tau)    


    "DEEP Q Learning "
    #This method need revisit since the input states details for each agents have to handle right
    def learn_v1(self, experiences, gamma,tau):
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
        actions_next = self.target_act(next_states)
        ##actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
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
        self.soft_update(self.critic_local, self.critic_target, tau)
        self.soft_update(self.actor_local, self.actor_target, tau)                     

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
    
    "When loading trained weights - copy ttarget netwrok also the same"
    def hard_update(self,local_model,target_model):
        for target_param,local_param in zip(target_model.parameters(),local_model.parameters()):
            target_param.data.copy_(local_param.data)

     #https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L199
    def load_trained_model(self, actor_path, critic_path):
        print('Loading trained models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor_local.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic_local.load_state_dict(torch.load(critic_path))
        print ('\r DONE')
    
    
    "load both target and local Actor Critic Network"           
    def load_trained_model(self, actor_path, critic_path,actor_tar_path,critic_tar_path):
        print('Loading trained models from {} and {}'.format(actor_path, critic_path,actor_tar_path,critic_tar_path))
        if actor_path is not None:
            self.actor_local.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic_local.load_state_dict(torch.load(critic_path))
        if actor_tar_path is not None:
            self.actor_target.load_state_dict(torch.load(actor_tar_path))
        if critic_tar_path is not None: 
            self.critic_target.load_state_dict(torch.load(critic_tar_path))
        print ('\r DONE')

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.17, sigma=0.24):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        #self.seed = random.seed(seed)
        self.reset()
        print("OUNoise params - Mu:{} , theta:{} sigma:{} ".format(self.mu,self.theta,self.sigma))
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        #Standard normal distribution for the action size
        dx = self.theta * (self.mu - x) + self.sigma *  np.random.randn(self.size)
        self.state = x + dx
        return self.state  

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size,num_agents,seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.num_agents = num_agents
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = [torch.from_numpy(np.vstack([e.state[iLoop] for e in experiences if e is not None])).float().to(device) for iLoop in range(self.num_agents)]
        actions = [torch.from_numpy(np.vstack([e.action[iLoop] for e in experiences if e is not None])).float().to(device) for iLoop in range(self.num_agents)]
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = [torch.from_numpy(np.vstack([e.next_state[iLoop] for e in experiences if e is not None])).float().to(device) for iLoop in range(self.num_agents)]
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
