import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

MODEL_SAVE_DIR = "./models"

class NeuralNetwork_Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(11,256)
        self.linear2 = nn.Linear(256,3)
    
    def forward(self, x):
        x = F.relu(self.linear1(x)) # 1st layer with tanh activation function
        #x = F.softmax(self.linear2(x)) # 2st layer with softmax activation function, ensures the outputs are probabilities
        x = self.linear2(x)
        return x
    
    def save(self, filename = "snakeAI.pth"):
        if not os.path.exists(MODEL_SAVE_DIR):
            os.makedirs(MODEL_SAVE_DIR)
        
        torch.save(self.state_dict, MODEL_SAVE_DIR + "/" + filename)


class Trainer_Q:
    def __init__(self, model, learning_rate, gamma):
        self.model = model
        self.learning_rate = learning_rate
        self.gamma = gamma 
        self.optimiser = optim.Adam(model.parameters(),lr=self.learning_rate)
        #self.loss = nn.CrossEntropyLoss()
        self.loss = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state= torch.tensor(next_state, dtype=torch.float)

        # if short term memory, then the state is a single row. We need inputs to be a list of rows
        #[1,2,3] -> [[1,2,3]]
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            game_over = (game_over,)
        
        # 1. Q_state0_current 
        Q_current = self.model(state)
        Q_new = Q_current.clone()

        for idx in range(len(game_over)):
            max_Q_val = reward[idx]

            if not game_over[idx]:
                max_Q_val = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            Q_new[idx][torch.argmax(action[idx]).item()] = max_Q_val


        self.optimiser.zero_grad()
        loss = self.loss(Q_new,Q_current)
        loss.backward()

        self.optimiser.step()



        

