import unittest
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

class neural_network(nn.Module):
    def __init__(self):
        """
        Initialize network layers
        """
        super(neural_network, self).__init__()

        self.pool = nn.MaxPool2d((1, 2), (1, 2))

        self.conv1 = nn.Conv2d(1, 25, (5, 5))
        self.conv1bn = nn.BatchNorm2d(25)

        self.conv2 = nn.Conv2d(25, 25, (1, 5))
        self.conv2bn = nn.BatchNorm2d(25)

        self.conv3 = nn.Conv2d(25, 25, (1, 5))
        self.conv3bn = nn.BatchNorm2d(25)

        self.conv4 = nn.Conv2d(25, 100, (1, 5))
        self.conv4bn = nn.BatchNorm2d(100)

        self.conv5 = nn.Conv2d(100, 10, (1, 1))
        self.conv5bn = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(4660, 466)
        self.fc1bn = nn.BatchNorm1d(466)

        self.fc2 = nn.Linear(466, 25)
        self.fc2bn = nn.BatchNorm1d(25)

        self.fc3 = nn.Linear(25, 4)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1bn(self.conv1(x))))
        logging.debug(x.size())

        x = self.pool(F.relu(self.conv2bn(self.conv2(x))))
        logging.debug(x.size())

        x = self.pool(F.relu(self.conv3bn(self.conv3(x))))
        logging.debug(x.size())

        x = self.pool(F.relu(self.conv4bn(self.conv4(x))))
        logging.debug(x.size())

        x = self.pool(F.relu(self.conv5bn(self.conv5(x))))
        logging.debug(x.size())

        x=x.view(-1,466*10)
        x = F.dropout(x, training=self.training)
        logging.debug(x.size())

        x = F.relu(self.fc1bn(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        logging.debug(x.size())

        x = F.relu(self.fc2bn(self.fc2(x)))
        x = F.dropout(x, training=self.training)

        x = self.fc3(x)
        y = F.softmax(x,dim=1)
        logging.debug(x.size())
        return x,y



class TestNet(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        self.net = neural_network()
        self.test_data_0 = Variable(torch.rand(1, 1, 1, 15000)).float()
        self.test_data_1 = Variable(torch.rand(200, 1, 1, 15000)).float()
        self.test_data_2 = Variable(torch.rand(200, 1, 5, 15000)).float()

    def test_forward_0(self):
        self.net.eval()
        self.net.forward(self.test_data_0)

    def test_forward_1(self):
        self.net.train()
        self.net.forward(self.test_data_1)

    def test_forward_2(self):
        self.net.train()
        self.net.forward(self.test_data_2)


