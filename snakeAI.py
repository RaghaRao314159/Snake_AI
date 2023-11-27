import pygame
from enum import Enum
from collections import namedtuple
import random
import numpy as np



pygame.init()
FONT = pygame.font.Font("Raleway-Regular.ttf", 25)

BLOCK_SIZE = 20
FPS = 80

# Top left hand corner or the screen is 0,0.
# Top left hand corner of a block is repesented by Coordinate tuple (immutable)
Coordinate = namedtuple("Coordinate", ['x', 'y'])

# Define direction
class Direction(Enum):
    #clockwise
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

    def next(self):
        next_value = (self.value + 1) % 4
        return Direction(next_value)

    def previous(self):
        previous_value = (self.value - 1) % 4
        return Direction(previous_value)


# Define colours
BACKGROUND = (16, 16, 16)
SNAKE = (8, 8, 232)
FOOD = (232, 8, 8)
TEXT = (255, 255, 255)
EYES = (0,0,0)


class snakeAI_game():
    def __init__(self, width = 15*BLOCK_SIZE, height=15*BLOCK_SIZE) -> None:
        self.width = width
        self.height = height

        # initialise display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("iSnake")
        self.clock = pygame.time.Clock()

        # initialise game state
        self.reset()
    
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Coordinate(BLOCK_SIZE*(self.width//(2*BLOCK_SIZE)), BLOCK_SIZE*(self.height//(2*BLOCK_SIZE)))
        self.snake = [self.head, Coordinate(self.head.x - BLOCK_SIZE, self.head.y), Coordinate(self.head.x - 2*BLOCK_SIZE, self.head.y)]

        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self.number_steps = 0

        self._generate_food()


    def _generate_food(self):
        x = random.randint(0, self.width//BLOCK_SIZE -1)*BLOCK_SIZE
        y = random.randint(0, self.height//BLOCK_SIZE -1)*BLOCK_SIZE
        self.food = Coordinate(x,y)
        if self.food in self.snake:
            self._generate_food()

    
    def play_step(self, action):
        self.frame_iteration += 1
        self.number_steps += 1

        # check if user wants to quit the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                pygame.quit() # exits pygame
                quit() # exits python program


        # change direction based on action
        # [1,0,0] -> continue, [0,1,0] -> turn right, [0,0,1] -> turn left
        
        
        if action[1] == 1:
            self.direction = self.direction.next()
        elif action[2] == 1:
            self.direction = self.direction.previous()


        # Add block in direction of movement
        self._move()

        # check if game is over
        reward = -0.1 #for every step, a reward is returned. {-0.02: one move made, 10: food eaten, -10: lose game}
        game_over = False

        if self.is_collision() or self.frame_iteration > 100*len(self.snake): # ensure game is not too long 
            game_over = True
            reward = -10
            #print("die: ", self.number_steps)
            self.number_steps = 0
            return reward, game_over, self.score
        
        # generate new food (add score) or move (remove end of snake)
        if self.head == self.food:
            self._generate_food()
            self.score += 1
            reward = 10
            #print("food: ", self.number_steps)
            self.number_steps = 0
        else:
            self.snake.pop()
        

        # update user interface
        self._update_user_interface()
        self.clock.tick(FPS)
        # clock 

        return reward, game_over, self.score

    def is_collision(self, point=None):
        if point == None:
            point = self.head
        # hits the boundary
        if (point.x > self.width - BLOCK_SIZE) or (point.x < 0) or (point.y > self.height - BLOCK_SIZE) or (point.y < 0):
            return True
        
        # hits itsef
        if point in self.snake[1:]:
            return True
        
        return False
    
    
    def _move(self):
        x,y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        
        self.head = Coordinate(x,y)
        self.snake.insert(0,self.head)
    
    
    def _update_user_interface(self):
        self.display.fill(BACKGROUND)

        for block in self.snake:
            #pygame.draw.rect(self.display, (0,0,255), pygame.Rect(block.x, block.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, SNAKE, pygame.Rect(block.x +1, block.y +1, BLOCK_SIZE-2, BLOCK_SIZE-2 ),border_radius=2)
        
        # draw eyes
        if self.direction == Direction.RIGHT or self.direction == Direction.LEFT:
            pygame.draw.rect(self.display, EYES, pygame.Rect(self.head.x + BLOCK_SIZE//2, self.head.y +5, 4, 4))
            pygame.draw.rect(self.display, EYES, pygame.Rect(self.head.x + BLOCK_SIZE//2, self.head.y + BLOCK_SIZE -9, 4, 4))
        
        elif self.direction == Direction.DOWN or self.direction == Direction.UP:
            pygame.draw.rect(self.display, EYES, pygame.Rect(self.head.x + 5, self.head.y + BLOCK_SIZE//2, 4, 4))
            pygame.draw.rect(self.display, EYES, pygame.Rect(self.head.x + BLOCK_SIZE -9, self.head.y + BLOCK_SIZE//2, 4, 4))
        
        pygame.draw.rect(self.display, FOOD, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),border_radius=2)

        text_score = FONT.render(f"Score: {self.score}", True, TEXT)
        self.display.blit(text_score, [0,0])

        pygame.display.flip()
    

