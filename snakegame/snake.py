import pygame
from enum import Enum
from collections import namedtuple
import random

pygame.init()
FONT = pygame.font.Font("Raleway-Regular.ttf", 25)

BLOCK_SIZE = 20
FPS = 10

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

# Define colours
BACKGROUND = (16, 16, 16)
SNAKE = (8, 8, 232)
FOOD = (232, 8, 8)
TEXT = (255, 255, 255)
EYES = (0,0,0)


class snake_game():
    def __init__(self, width = 30*BLOCK_SIZE, height=30*BLOCK_SIZE) -> None:
        self.width = width
        self.height = height

        # initialise display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("iSnake")
        self.clock = pygame.time.Clock()

        # initialise game state
        self.direction = Direction.RIGHT
        self.head = Coordinate(self.width/2, self.height/2)
        self.snake = [self.head, Coordinate(self.head.x - BLOCK_SIZE, self.head.y), Coordinate(self.head.x - 2*BLOCK_SIZE, self.head.y)]

        self.score = 0
        self.food = None

        self._generate_food()

    def _generate_food(self):
        x = random.randint(0, self.width//BLOCK_SIZE -1)*BLOCK_SIZE
        y = random.randint(0, self.height//BLOCK_SIZE -1)*BLOCK_SIZE
        self.food = Coordinate(x,y)
        if self.food in self.snake:
            self._generate_food()

    
    def play_step(self):
        # collect user input

        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                pygame.quit() # exits pygame
                quit() # exits python program

            if event.type == pygame.KEYDOWN:
                if (event.key == pygame.K_RIGHT) and (self.direction != Direction.LEFT):
                    self.direction = Direction.RIGHT

                elif (event.key == pygame.K_DOWN) and (self.direction != Direction.UP):
                    self.direction = Direction.DOWN

                elif (event.key == pygame.K_LEFT) and (self.direction != Direction.RIGHT):
                    self.direction = Direction.LEFT

                elif (event.key == pygame.K_UP) and (self.direction != Direction.DOWN):
                    self.direction = Direction.UP
                
                break

        # Add block in direction of movement
        self._move()

        # check if game is over
        game_over = False

        if self._is_collision():
            game_over = True
            return game_over, self.score
        
        # generate new food (add score) or move (remove end of snake)
        if self.head == self.food:
            self._generate_food()
            self.score += 1
        else:
            self.snake.pop()
            

        # update user interface
        self._update_user_interface()
        self.clock.tick(FPS)
        # clock 

        return game_over, self.score

    def _is_collision(self):
        # hits the boundary
        if (self.head.x > self.width - BLOCK_SIZE) or (self.head.x < 0) or (self.head.y > self.height - BLOCK_SIZE) or (self.head.y < 0):
            return True
        
        # hits itsef
        if self.head in self.snake[1:]:
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
    


if __name__ == "__main__":

    # start game

    game = snake_game()

    while True:
        game_over, score = game.play_step()

        # break if game is over
        if game_over:
            break


    
    print('Final score:', score)
    
    pygame.quit()
