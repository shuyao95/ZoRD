import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Model:
    def __init__(self, device):
        self.device = device
        self.model = Net().to(device)
        self.model.load_state_dict(torch.load('attack/mnist_cnn.pt', map_location=device))
        self.model.eval()

    def get_loss_and_grad(self, image, label, targeted=True, get_grad=True):
        image = image.clone()
        if get_grad:
            image.requires_grad_()
        input_image = (image - 0.1307) / 0.3081
        output = self.model(input_image)
        label_term = output[:, label]
        other = output + 0.0
        other[:, label] = -1e8
        other_term = torch.max(other, dim=1).values
        if targeted:
            loss = label_term - other_term
        else:
            loss = other_term - label_term
        
        loss = torch.squeeze(loss)
        if get_grad:
            loss.backward()
            grad = image.grad
            return loss.detach().cpu().numpy(), grad.detach()
        else:
            return loss.detach().cpu().numpy()

    def get_pred(self, image):
        input_image = (image - 0.1307) / 0.3081
        output = self.model(input_image).detach().cpu().numpy().squeeze()
        return np.argmax(output)