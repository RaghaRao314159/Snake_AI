import torch
import random 
import numpy as np
from collections import deque
from snakeAI import snakeAI_game, Direction, Coordinate, BLOCK_SIZE
from model import NeuralNetwork_Q, Trainer_Q
from plot import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
LIMIT_OF_EXPLOITATION = 80 # below this many number of games, we want the AI to randomly pick an action

class Agent:
    def __init__(self):
        self.number_games = 0
        self.epsilon = 0 # randomness factor. Initially we want to explore (random). Then we want to exploit (deterministic).
        self.gamma = 0.9 # discount rate. How much will future states affect current states.
        self.memory = deque(maxlen=MAX_MEMORY)  # pop left if memory exceeded

        self.model = NeuralNetwork_Q() # TODO
        self.trainer = Trainer_Q(self.model, LEARNING_RATE, self.gamma) # TODO

        #model, trainer
        pass

    
    def get_state(self, snakeAI_game):
        danger_straight, danger_right, danger_left = False, False, False,
        direction_right, direction_down, direction_left, direction_up =  snakeAI_game.direction == Direction.RIGHT, snakeAI_game.direction == Direction.DOWN, snakeAI_game.direction == Direction.LEFT, snakeAI_game.direction == Direction.UP

        current_head = snakeAI_game.head
        right_neighbour = Coordinate(current_head.x + BLOCK_SIZE, current_head.y)
        down_neighbour = Coordinate(current_head.x, current_head.y + BLOCK_SIZE)
        left_neighbour = Coordinate(current_head.x - BLOCK_SIZE, current_head.y)
        up_neighbour  =  Coordinate(current_head.x, current_head.y - BLOCK_SIZE)
        
        if snakeAI_game.is_collision(right_neighbour):
            if direction_right:
                danger_straight = True
            elif direction_up:
                danger_right = True
            elif direction_down:
                danger_left = True
        
        if snakeAI_game.is_collision(left_neighbour):
            if direction_left:
                danger_straight = True
            elif direction_down:
                danger_right = True
            elif direction_up:
                danger_left = True
        
        if snakeAI_game.is_collision(up_neighbour):
            if direction_up:
                danger_straight = True
            elif direction_left:
                danger_right = True
            elif direction_right:
                danger_left = True

        if snakeAI_game.is_collision(down_neighbour):
            if direction_down:
                danger_straight = True
            elif direction_right:
                danger_right = True
            elif direction_left:
                danger_left = True

        current_food = snakeAI_game.food
        food_right, food_down, food_left, food_up = current_food.x > current_head.x, current_food.y > current_head.y, current_food.x < current_head.x, current_food.y < current_head.y

        return np.array([danger_straight, danger_right, danger_left, direction_right, direction_down, direction_left, direction_up, food_right, food_down, food_left, food_up], dtype=int)


    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))
        pass

    def train_long_term_memory(self):
        if len(self.memory) >  BATCH_SIZE:
            training_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            training_sample = self.memory
        
        states, actions, rewards, next_states, game_overs = zip(*training_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_term_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        action = [0,0,0] 
        #self.epsilon = LIMIT_OF_EXPLOITATION - self.number_games
        self.epsilon = LIMIT_OF_EXPLOITATION - self.number_games
        if random.randint(0,2*LIMIT_OF_EXPLOITATION) < self.epsilon:
            random_action_idx = random.randint(0,2)
            action[random_action_idx] = 1
        else: 
            model_input = torch.tensor(state, dtype=torch.float)
            prediction = self.model(model_input)
            # alternatively: predicted_action_idx = np.argmax(prediction).item()
            predicted_action_idx = torch.argmax(prediction).item()
            action[predicted_action_idx] = 1
        return action


def train(file):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = snakeAI_game()

    while True:
        # get current state
        current_state = agent.get_state(game)

        # get next move
        next_move = agent.get_action(current_state)

        # perform move and get next state
        reward, game_over, score = game.play_step(next_move)
        next_state = agent.get_state(game)
        
        #train single step
        agent.train_short_term_memory(current_state, next_move, reward, next_state, game_over)

        # add to training dataset
        agent.remember(current_state, next_move, reward, next_state, game_over)

        if game_over:
            game.reset()
            agent.number_games += 1

            # train long memory
            agent.train_long_term_memory()

            # update record
            if score > record:
                record = score
                agent.model.save()
            
            file.write(f"Game {agent.number_games}  Score: {score}  Record: {record} \n")

            # plot
            total_score += score
            plot_scores.append(score)
            plot_mean_scores.append(total_score/agent.number_games)

            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    file = open("results.txt", "w")
    train(file)
    #file.close()

 