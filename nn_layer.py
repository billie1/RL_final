import torch.nn as nn
from torchvision import models


class PPOActor(nn.Module):
    def __init__(self, num_classes):
        super(PPOActor, self).__init__()

        self.actor_model = models.resnet50(pretrained=False)

        for param in self.actor_model.parameters():
            param.requires_grad = False

        num_actor_features = self.actor_model.fc.in_features
        self.actor_model.fc = nn.Linear(num_actor_features, num_classes)
        self.action_output = nn.Softmax(dim=1)

    def forward(self, x):
        action = self.actor_model(x)
        action = self.action_output(action)
        return action


class PPOCritic(nn.Module):
    def __init__(self):
        super(PPOCritic, self).__init__()

        self.critic_model = models.resnet50(pretrained=False)

        for param in self.critic_model.parameters():
            param.requires_grad = False

        num_critic_features = self.critic_model.fc.in_features
        self.critic_model.fc = nn.Linear(num_critic_features, 1)

    def forward(self, x):
        state_value = self.critic_model(x)
        return state_value
