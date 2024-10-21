import numpy as np
import pygame
import random
import sys

random.seed()

# Constants
Dimensions = 10
Sqaure_size = 40
Screen_size = Dimensions * Sqaure_size

Speed = 400

# Color
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (2, 100, 64)
RED = (255, 8, 0)

# Initialize Pygame
pygame.init()
game_screen = pygame.display.set_mode((Screen_size, Screen_size))
pygame.display.set_caption('Snake Game')


# Define SnakeGame    
class SnakeGame:
    def __init__(self, screen, game_size):
        self.step_count = 0
        self.screen = screen
        self.size = game_size
        self.move_timer = pygame.time.get_ticks()
        self.snake_tiles, self.direction = self.generate_snake()
        self.apple_tile = self.generate_random_apple()

    @property
    def snake_len(self):
        return len(self.snake_tiles)

    def generate_snake(self):
        snake = [[random.randint(0,self.size-1),
                  random.randint(1,self.size-2)]]
        
        if snake[0][1] < 5:
            dir = 2
            snake.append([snake[0][0], snake[0][1]-1])
        else:
            dir = 0
            snake.append([snake[0][0], snake[0][1]+1])

        return snake, dir

    def generate_random_apple(self):
        while True:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if [x, y] not in self.snake_tiles:
                return [x, y]

    def move_snake(self):
        vec = {0:[0, -1],
               1:[1, 0],
               2:[0, 1],
               3:[-1, 0]}

        dir_add = vec[self.direction]

        head = [x + y for x, y in zip(self.snake_tiles[0], dir_add)]
        
        # Check for death   
        if head in self.snake_tiles[1:]:
            return True
        elif head[0] > 9 or head[0] < 0 or head[1] > 9 or head[1] < 0:
            return True
            
        self.snake_tiles = [head] + self.snake_tiles

        # Check for growth        
        if head == self.apple_tile:
            self.apple_tile = self.generate_random_apple()           
        else:
            self.snake_tiles = self.snake_tiles[:-1]

        self.draw_game()
    
    def draw_game(self):
        for row in range(self.size):
            for col in range(self.size):
                rectangle = pygame.Rect(row * Sqaure_size, col * Sqaure_size,
                                        Sqaure_size, Sqaure_size)
                pygame.draw.rect(self.screen, BLACK, rectangle)
                rectangle = pygame.Rect(row * Sqaure_size + 2, col * Sqaure_size + 2,
                                        Sqaure_size - 4, Sqaure_size - 4)
                pygame.draw.rect(self.screen, WHITE, rectangle)

        for tile in self.snake_tiles:
                rectangle = pygame.Rect(tile[0] * Sqaure_size + 2, tile[1] * Sqaure_size + 2,
                                        Sqaure_size - 4, Sqaure_size - 4)
                pygame.draw.rect(self.screen, GREEN, rectangle)

        head = self.snake_tiles[0]

        if self.direction % 2 ==0:
            pygame.draw.circle(self.screen, BLACK, [head[0] * Sqaure_size + 10, 
                                                    (head[1] + 1/4 * self.direction) * Sqaure_size + 10], 5)
            pygame.draw.circle(self.screen, BLACK, [head[0] * Sqaure_size + 30, 
                                                    (head[1] + 1/4 * self.direction) * Sqaure_size + 10], 5)
        else:
            pygame.draw.circle(self.screen, BLACK, [(head[0] - 1/4 * (self.direction - 3)) * Sqaure_size + 10, 
                                                    head[1] * Sqaure_size + 10], 5)
            pygame.draw.circle(self.screen, BLACK, [(head[0] - 1/4 * (self.direction - 3)) * Sqaure_size + 10, 
                                                    head[1] * Sqaure_size + 30], 5)
        
        rectangle = pygame.Rect(self.apple_tile[0] * Sqaure_size + 2, self.apple_tile[1] * Sqaure_size + 2,
                                Sqaure_size - 4, Sqaure_size - 4)
        pygame.draw.rect(self.screen, RED, rectangle)

        pygame.display.update()

game = SnakeGame(game_screen, Dimensions)
game.draw_game()

# Main game loop
game_over = False
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_LEFT:
                game.direction = 3
            elif event.key == pygame.K_RIGHT:
                game.direction = 1
            elif event.key == pygame.K_UP:
                game.direction = 0
            elif event.key == pygame.K_DOWN:
                game.direction = 2
            
            # if event.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
            #     game.move_timer = pygame.time.get_ticks()
            #     game.move_snake()


    current_time = pygame.time.get_ticks()
    if current_time - game.move_timer >= Speed:
        game_over = game.move_snake()
        game.move_timer = current_time

        if game_over:
            print(game.snake_len)
            pygame.quit()
            sys.exit()       
