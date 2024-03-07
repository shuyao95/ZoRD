import numpy as np
import jax.numpy as jnp

import torch
from torchvision import datasets, transforms
from attack.mnist_cnn import Model as MNIST_Model
from attack.cifar10_cnn import Model as CIFAR10_Model
from utils import *

class MNIST_Attack:
    def __init__(self, dim=784, lb=-0.3, ub=0.3):
        self.lb = lb * jnp.ones(dim)
        self.ub = ub * jnp.ones(dim)
        
        dataset = datasets.MNIST('data', download=True, train=False, transform=transforms.ToTensor())
        self.size = (1, 1, 28, 28)
        self.dim = 28 * 28
        
        self.device = torch.device("cuda")
        self.model = MNIST_Model(self.device)
        self.data, self.target = dataset[1]
        
    def __call__(self, x):
        x = unnormalize(x, self.lb, self.ub, SCALE)
        x = np.array(x)
        
        new_image = torch.from_numpy(x.reshape(*self.size)).float() + self.data
        new_image = torch.clamp(new_image, 0, 1)
        
        target_label = (self.target + 1) % 10
        
        loss = self.model.get_loss_and_grad(new_image.to(self.device), target_label, targeted=True, get_grad=False)
        
        return - loss
    
    
class CIFAR10_Attack:
    def __init__(self, dim=1024, lb=-0.1, ub=0.1):
        self.lb = lb * jnp.ones(dim)
        self.ub = ub * jnp.ones(dim)
        
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
        self.size = (1, 1, 32, 32)
        self.dim = 1 * 32 * 32
        
        self.trans_lb = (np.zeros(self.size) - np.array(mean).reshape(1, 3, 1, 1)) / np.array(std).reshape(1, 3, 1, 1)
        self.trans_ub = (np.ones(self.size) - np.array(mean).reshape(1, 3, 1, 1)) / np.array(std).reshape(1, 3, 1, 1)
        
        self.device = torch.device("cuda")
        self.model = CIFAR10_Model(self.device)
        self.data, self.target = dataset[1]
        
    def __call__(self, x):
        x = unnormalize(x, self.lb, self.ub, SCALE)
        x = np.array(x)
        
        new_image = torch.from_numpy(x.reshape(*self.size)).float() + self.data
        new_image = torch.clamp(new_image, torch.from_numpy(self.trans_lb), torch.from_numpy(self.trans_ub))
        
        target_label = (self.target + 1) % 10
        
        loss = self.model.get_loss_and_grad(new_image.to(self.device), target_label, targeted=True, get_grad=False)
        
        return - loss
        