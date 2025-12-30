import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# torch.manual_seed(42)
class PolicyNet(nn.Module):
    def __init__(self, temperature=2.0):
        super().__init__()
        self.fc1 = nn.Linear(4, 5)  # 2 inputs, 5 neurons in hidden layer
        self.fc2 = nn.Linear(5, 5)  # 5 neurons to 5 neuron in second hidden layer
        self.fc3 = nn.Linear(5, 2)  # 2 output neurons
        self.temperature = temperature

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation after the first layer
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        return x   #position 0 is action 0, position 1 is action 1
    
class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 5)  # 5 inputs, 5 neurons in hidden layer
        self.fc2 = nn.Linear(5, 5)  # 5 neurons to 1 output neuron
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation after the first layer
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)
        return x

def policy_initialize():
    Policy = PolicyNet()
    print(f"last layer weights is {Policy.fc3.weight}")
    #tensor([[-0.1800,  0.2867, -0.3658,  0.2302,  0.4043],
        #[ 0.1898, -0.3804,  0.4114, -0.1021, -0.1005]], requires_grad=True)
    torch.save(Policy.state_dict(), 'Weights/Policy_Weights.pth')

def advantage_initialize():
    Advantage = ValueNet()
    print(f"last layer weights is {Advantage.fc3.weight}")
    #last layer weights is Parameter containing:
    #tensor([[-0.3165,  0.0360, -0.3769, -0.3575,  0.2984]], requires_grad=True)
    torch.save(Advantage.state_dict(), 'Weights/Value_Weights.pth')


def main():
   # Create an instance of the model
   policy_initialize()
   advantage_initialize()
   

if __name__ == "__main__":
    main()

