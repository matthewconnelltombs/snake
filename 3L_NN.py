import numpy as np
import csv
import pygame
import random
import pickle

random.seed()

# Constants
DIMENSIONS = 8
SQUARE_SIZE = 40
SCREEN_SIZE = DIMENSIONS * SQUARE_SIZE

SPEED = 0.1

INPUT_SIZE = 8*3 # + 4
LAYER1_SIZE = 16
LAYER2_SIZE = 8
OUTPUT_SIZE = 3
POPULATION_SIZE = 200

# Color
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (2, 100, 64)
RED = (255, 8, 0)

# Initialize Pygame
pygame.init()
GAME_SCREEN = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption('Snake Game')


# Define SnakeGame    
class SnakeGame:
    def __init__(self, screen, game_size):
        self.step_count = 0
        self.health = game_size ** 2
        self.screen = screen
        self.size = game_size
        self.move_timer = pygame.time.get_ticks()
        self.snake_tiles, self.direction = self.generate_snake()
        self.apple_tile = self.generate_random_apple()

    @property
    def snake_len(self):
        return len(self.snake_tiles)

    # Create snake
    def generate_snake(self):
        snake = [[random.randint(1,self.size-2),
                  random.randint(1,self.size-2)]]
        
        if snake[0][1] < 5:
            dir = 2
            snake.append([snake[0][0], snake[0][1]-1])
        else:
            dir = 0
            snake.append([snake[0][0], snake[0][1]+1])

        return snake, dir

    # Create apple
    def generate_random_apple(self):
        while True:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if [x, y] not in self.snake_tiles:
                return [x, y]

    # Move snake based on NN predict
    def move_snake(self, NN):

        vec = {0:[0, -1],
               0.5:[1,-1],
               1:[1, 0],
               1.5:[1,1],
               2:[0, 1],
               2.5:[-1,1],
               3:[-1, 0],
               3.5:[-1,-1]}        

        head = self.snake_tiles[0]

        # measure wall distance
        wall = [head[1], min(head[1], self.size - 1 - head[0]),
                self.size - 1 - head[0], min(self.size - 1 - head[0], self.size - 1 - head[1]),
                self.size - 1 - head[1], min(self.size - 1 - head[1], head[0]),
                head[0], min(head[0], head[1]),]

        wall_scale = [1/(x+1) for x in wall]

        # Measure snake distance
        snake = []

        for dir in range(8):
            counter = 0
            check_tile = head
            while True:
                check_tile = [x + y for x, y in zip(check_tile, vec[dir/2])]
                
                if check_tile in self.snake_tiles:
                    snake.append(1)
                    break
                elif counter == wall[dir]:
                    snake.append(0)
                    break

                counter += 1
            
        # Measure apple distance
        apple = []

        for dir in range(8):
            counter = 0
            check_tile = head
            while True:
                check_tile = [x + y for x, y in zip(check_tile, vec[dir/2])]
                
                if check_tile == self.apple_tile:
                    apple.append(1)
                    break
                elif counter == wall[dir]:
                    apple.append(0)
                    break
                
                counter += 1

        # dir_vec = [0] * 4
        # dir_vec[self.direction] = 1

        # Input vector
        if OUTPUT_SIZE == 3:
            vector = (wall_scale[2*self.direction:] + wall_scale[:2*self.direction] + 
                    snake[2*self.direction:] + snake[:2*self.direction] + 
                    apple[2*self.direction:] + apple[:2*self.direction] 
                    # + dir_vec
                    )

            change_dir = NN.predict(vector) - 1
            self.direction = (self.direction + change_dir) % 4

        elif OUTPUT_SIZE == 4:
            vector = wall_scale + snake + apple # + dir_vec
            self.direction = NN.predict(vector)

        # Make move
        dir_add = vec[self.direction]
        head = [x + y for x, y in zip(head, dir_add)]
        
        # Check for death
        if head in self.snake_tiles[1:]:
            return True
        elif head[0] > self.size - 1 or head[0] < 0 or head[1] > self.size - 1 or head[1] < 0:
            return True
            
        self.step_count += 1

        self.snake_tiles = [head] + self.snake_tiles

        # Check for growth        
        if head == self.apple_tile:
            self.health = self.size ** 2
            self.apple_tile = self.generate_random_apple()           
        else:
            self.health -= 1
            self.snake_tiles = self.snake_tiles[:-1]

        self.draw_game()
    
    # Draw the game
    def draw_game(self):
        # Draw board
        for row in range(self.size):
            for col in range(self.size):
                rectangle = pygame.Rect(row * SQUARE_SIZE, col * SQUARE_SIZE,
                                        SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.screen, BLACK, rectangle)
                rectangle = pygame.Rect(row * SQUARE_SIZE + 2, col * SQUARE_SIZE + 2,
                                        SQUARE_SIZE - 4, SQUARE_SIZE - 4)
                pygame.draw.rect(self.screen, WHITE, rectangle)

        # Draw snake
        for tile in self.snake_tiles:
                rectangle = pygame.Rect(tile[0] * SQUARE_SIZE + 2, tile[1] * SQUARE_SIZE + 2,
                                        SQUARE_SIZE - 4, SQUARE_SIZE - 4)
                pygame.draw.rect(self.screen, GREEN, rectangle)

        head = self.snake_tiles[0]

        if self.direction % 2 ==0:
            pygame.draw.circle(self.screen, BLACK, [head[0] * SQUARE_SIZE + 10, 
                                                    (head[1] + 1/4 * self.direction) * SQUARE_SIZE + 10], 5)
            pygame.draw.circle(self.screen, BLACK, [head[0] * SQUARE_SIZE + 30, 
                                                    (head[1] + 1/4 * self.direction) * SQUARE_SIZE + 10], 5)
        else:
            pygame.draw.circle(self.screen, BLACK, [(head[0] - 1/4 * (self.direction - 3)) * SQUARE_SIZE + 10, 
                                                    head[1] * SQUARE_SIZE + 10], 5)
            pygame.draw.circle(self.screen, BLACK, [(head[0] - 1/4 * (self.direction - 3)) * SQUARE_SIZE + 10, 
                                                    head[1] * SQUARE_SIZE + 30], 5)
        
        # Draw fruit
        rectangle = pygame.Rect(self.apple_tile[0] * SQUARE_SIZE + 2, self.apple_tile[1] * SQUARE_SIZE + 2,
                                SQUARE_SIZE - 4, SQUARE_SIZE - 4)
        pygame.draw.rect(self.screen, RED, rectangle)

        pygame.display.update()


class NeuralNetwork:
    def __init__(self):
        self.weights_input_layer1 = np.random.randn(INPUT_SIZE, LAYER1_SIZE)
        self.weights_layer1_layer2 = np.random.randn(LAYER1_SIZE, LAYER2_SIZE)
        self.weights_layer2_output = np.random.randn(LAYER2_SIZE, OUTPUT_SIZE)
        self.bias_input = np.zeros(INPUT_SIZE)
        self.bias_layer = np.zeros(LAYER2_SIZE)
        self.bias_output = np.zeros(OUTPUT_SIZE)

    def predict(self, inputs):
        hidden_layer1 = np.dot(inputs + self.bias_input, self.weights_input_layer1)
        hidden_layer1_activation = self.relu(hidden_layer1)

        hidden_layer2 = np.dot(hidden_layer1_activation, self.weights_layer1_layer2)
        hidden_layer2_activation = self.relu(hidden_layer2 + self.bias_layer)

        output_layer = np.dot(hidden_layer2_activation, self.weights_layer2_output)
        output = self.softmax(output_layer + self.bias_output)

        predicted_action = np.argmax(output)

        return predicted_action

    @staticmethod
    def relu(x):
        return  np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))  # Subtracting max(x) for numerical stability
        return exp_x / exp_x.sum(axis=0, keepdims=True)


def selection(fitness_scores, frac):
    # Select individuals based on their fitness scores
    selected_indices = np.random.choice(range(POPULATION_SIZE), POPULATION_SIZE // int(1/frac), 
                                        p=fitness_scores/np.sum(fitness_scores), replace=True)

    return selected_indices

def crossover(neural_networks, selected_indices, frac):
    selected_indices = np.append(selected_indices, selected_indices[0])
    offspring = []

    for i in range(0, POPULATION_SIZE // int(1/frac)):
        parent1 = neural_networks[selected_indices[i]]
        parent2 = neural_networks[selected_indices[i + 1]]

        for _ in range(int(1/frac) - 1):

            p1_genes = np.concatenate([parent1.weights_input_layer1.flatten(), 
                                       parent1.weights_layer1_layer2.flatten(),
                                       parent1.weights_layer2_output.flatten(),
                                       parent1.bias_input.flatten(),
                                       parent1.bias_layer.flatten(),
                                       parent1.bias_output.flatten()])
            p2_genes = np.concatenate([parent2.weights_input_layer1.flatten(), 
                                       parent2.weights_layer1_layer2.flatten(),
                                       parent2.weights_layer2_output.flatten(),
                                       parent2.bias_input.flatten(),
                                       parent2.bias_layer.flatten(),
                                       parent2.bias_output.flatten()])

            # mask = np.random.rand(len(p1_genes))

            # child_genes = np.where(mask < 0.5, p1_genes, p2_genes) 

            crossover_points = np.sort(np.random.choice(len(p1_genes), 2, replace=False))

            child_genes = p1_genes.copy()
            child_genes[crossover_points[0]:crossover_points[1]] = p2_genes[crossover_points[0]:crossover_points[1]]

            child = NeuralNetwork()
            
            child.weights_input_layer1 = child_genes[:INPUT_SIZE * LAYER1_SIZE].reshape((INPUT_SIZE, LAYER1_SIZE))
            child.weights_layer1_layer2 = child_genes[INPUT_SIZE * LAYER1_SIZE:(INPUT_SIZE+LAYER2_SIZE)*LAYER1_SIZE].reshape((LAYER1_SIZE, LAYER2_SIZE))
            child.weights_layer2_output = child_genes[(INPUT_SIZE+LAYER2_SIZE)*LAYER1_SIZE:(INPUT_SIZE+LAYER2_SIZE)*LAYER1_SIZE + LAYER2_SIZE*OUTPUT_SIZE].reshape((LAYER2_SIZE, OUTPUT_SIZE))

            child.bias_input = child_genes[(INPUT_SIZE+LAYER2_SIZE)*LAYER1_SIZE + LAYER2_SIZE*OUTPUT_SIZE:(INPUT_SIZE+LAYER2_SIZE)*LAYER1_SIZE + LAYER2_SIZE*OUTPUT_SIZE+INPUT_SIZE].reshape((INPUT_SIZE))
            child.bias_layer = child_genes[(INPUT_SIZE+LAYER2_SIZE)*LAYER1_SIZE + LAYER2_SIZE*OUTPUT_SIZE+INPUT_SIZE:(INPUT_SIZE+LAYER2_SIZE)*LAYER1_SIZE + LAYER2_SIZE*OUTPUT_SIZE+INPUT_SIZE+LAYER2_SIZE].reshape((LAYER2_SIZE))
            child.bias_output = child_genes[(INPUT_SIZE+LAYER2_SIZE)*LAYER1_SIZE + LAYER2_SIZE*OUTPUT_SIZE+INPUT_SIZE+LAYER2_SIZE:].reshape((OUTPUT_SIZE))

            offspring.extend([child])
        
        offspring.extend([neural_networks[selected_indices[i]]])

    return offspring

def mutation(offspring, gen):
    mutation_rate = max(0.2 - gen/2000, 0.05)
    mutation_weight = max(0.2 - gen/2000, 0.05)

    for child in offspring:
        # Apply mutations to the genes
        mutation_mask = np.random.rand(*child.weights_input_layer1.shape) < mutation_rate
        child.weights_input_layer1 += mutation_mask * np.random.randn(*child.weights_input_layer1.shape) * mutation_weight

        mutation_mask = np.random.rand(*child.weights_layer1_layer2.shape) < mutation_rate
        child.weights_layer1_layer2 += mutation_mask * np.random.randn(*child.weights_layer1_layer2.shape) * mutation_weight

        mutation_mask = np.random.rand(*child.weights_layer2_output.shape) < mutation_rate
        child.weights_layer2_output += mutation_mask * np.random.randn(*child.weights_layer2_output.shape) * mutation_weight

        mutation_mask = np.random.rand(*child.bias_input.shape) < mutation_rate
        child.bias_input += mutation_mask * np.random.randn(*child.bias_input.shape) * mutation_weight/10

        mutation_mask = np.random.rand(*child.bias_layer.shape) < mutation_rate
        child.bias_layer += mutation_mask * np.random.randn(*child.bias_layer.shape) * mutation_weight/10

        mutation_mask = np.random.rand(*child.bias_output.shape) < mutation_rate
        child.bias_output += mutation_mask * np.random.randn(*child.bias_output.shape) * mutation_weight/10

    return offspring

def fitness_function(snake_length, step_count, gen):
    return (step_count + 25 * (snake_length-2)**2 + 1.75**(snake_length-2) - (0.25 * step_count)**1.25)

def genetic_algorithm(neural_networks, fitness_scores, gen, frac):
    # Selection
    selected_indices = selection(fitness_scores, frac)
    # Crossover
    offspring = crossover(neural_networks, selected_indices, frac)
    # Mutation
    mutated_offspring = mutation(offspring, gen)

    return mutated_offspring

# File names
pickle_file = "3L/data_3L.pickle"
csv_file = "3L/results_3L.csv"

# Load population
try:
    with open(pickle_file, "rb") as f:
        population = pickle.load(f)
except Exception as ex:
    population = [NeuralNetwork() for _ in range(POPULATION_SIZE)]

# Main game loop
def play_game(NN):
    game = SnakeGame(GAME_SCREEN, DIMENSIONS)
    game.draw_game()

    game_over = False
    while True:
        # For viewing game
        if SPEED > 1:
            current_time = pygame.time.get_ticks()
            if current_time - game.move_timer >= SPEED:
                game_over = game.move_snake(NN)
                game.move_timer = current_time
                
        else:        
            game_over = game.move_snake(NN)
            # game.move_timer = current_time

        if game.health == 0 or game_over:
            return game.snake_len, game.step_count

# Trimmed mean for population fitness
def trimmed_mean(data, trim_percentage=0.1):
    sorted_data = np.sort(data)
    trim_size = int(len(sorted_data) * trim_percentage)
    trimmed_data = sorted_data[trim_size:-trim_size]
    mean = np.mean(trimmed_data)
    return mean

# Make File/Read Gen
try:
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        last_row = list(reader)[-1]
        Gen = int(last_row[0])
except FileNotFoundError:
    Gen = 0

# Training Loop
for iter in range(1000):
    pop_fitness = []
    pop_score = []
    for pop in population:
        length, count = play_game(pop)
        pop_fitness.append(fitness_function(length, count, Gen))
        pop_score.append(length-2)

    # Metrics
    Gen += 1
    T_Mean = round(trimmed_mean(pop_fitness))
    Min_Fit = round(min(pop_fitness))
    Max_Fit = round(max(pop_fitness))
    Max_Score = max(pop_score)

    print(f"Gen: {Gen}, T_Mean: {T_Mean}, Min: {Min_Fit}, Max: {Max_Fit}, Max Score: {Max_Score}")
    
    # Save metrics
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Generation", "T Fitness", "Min Fitness", "Max Fitness", "Max Score"])        
        writer.writerow([Gen, T_Mean, Min_Fit, Max_Fit, Max_Score])

    with open(pickle_file, "wb") as f:
        pickle.dump(population, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save every 100th GEN
    if Gen % 100 == 0:
        with open("3L/data_3L_" + str(Gen) + ".pickle", "wb") as f:
            pickle.dump(population, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    population = genetic_algorithm(population, pop_fitness, Gen, 0.25)


# Viewing best in Gen
# pop_fitness = []
# for pop in population:
#     random.seed(1)
#     length, count = play_game(pop)
#     pop_fitness.append(fitness_function(length, count, Gen))

# best = np.argmax(pop_fitness)
# print(best)

# top_indices = np.argsort(pop_fitness)[-3:]
# print(top_indices)

# best = 89

# random.seed(1)
# SPEED = 150
# length, count = play_game(population[best])
# print(length, count)
