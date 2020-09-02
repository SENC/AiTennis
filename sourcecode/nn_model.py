import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

"""Initialise Hidden Layer """
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

"""Actor (Policy) Model."""
class Actor(nn.Module):
   
    """Initialize Actor instance ="""
    def __init__(self, state_size, action_size, seed, fc1_units=500, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()
        
    """Re-set all input , hidden and output layers weights """
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    """Feed Forward Network arch with Relu activation function."""
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        """Used relu as a activation function for Neurons y=max(0,x) """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))

"""Critic (Value) Model."""
class Critic(nn.Module):
    
    """Initialize Critic (Action-Value) instance ="""
    
    def __init__(self, state_size, action_size, seed, fcs1_units=500, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        #self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fcs1 = nn.Linear((state_size+action_size)*2,fcs1_units)
        #Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
        self.batch = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        """randomly zeroes some of the elements of the input tesnor"
            This has proven to be an effective technique for regularization and
            preventing the co-adaptation of neurons as described in the paper
            `Improving neural networks by preventing co-adaptation of feature
            detectors`_ ."""
        self.dropout = nn.Dropout(p=0.03)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()
    
    """Reset all layers neuron's weight ="""
    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    """Network architecture with Relu activation function """
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        #xs = F.relu(self.fcs1(state))
        xs =torch.cat((state,action),dim=1)
        #x = torch.cat((xs, action), dim=1)
        #Activation
        x = F.relu(self.fcs1(xs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #his has proven to be an effective technique for regularization and 
        # preventing the co-adaptation of neurons
        x = self.dropout(x)
        return x
