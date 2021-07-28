import random
import os
import math
import pygame
import neat

from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    K_SPACE,
    K_RETURN,
    KEYDOWN,
    QUIT,
)


pygame.init()

#Screen size constants
HEIGHT = 800
WIDTH = 800

#Create window
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("dodgeNeat")

#Get font for displaying info
FONT = pygame.font.Font('freesansbold.ttf', 20)

class Player:
    SIZE = 80
    def __init__(self):
        #Set x and y to center bottom of screen
        self.x = WIDTH//2
        self.y = HEIGHT - self.SIZE//2

        #Player velocity
        self.vel = 10

        #Give each player a random colour to make them easier to differentiate
        self.colour = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))

        #Create surfaces and rectangles for player
        self.surf = pygame.Surface((self.SIZE, self.SIZE))
        self.surf.fill(self.colour)
        self.rect = pygame.Rect(self.x - self.SIZE // 2, self.y - self.SIZE // 2, self.SIZE, self.SIZE)

    #Check if player hits edge of screen and if so, remove them
    def off_screen(self, i):
        if self.rect.x < 0 or self.rect.x > WIDTH - self.SIZE:
            remove_player(i)

    #Move player based on output
    def move(self, dir):
        if dir == 0:
            self.rect.move_ip(-self.vel, 0)
        if dir == 1:
            self.rect.move_ip(self.vel, 0)
    
    #Check collisions between player and enemies
    def collide(self, enemy, i):
        if self.rect.colliderect(enemy.rrect) or self.rect.colliderect(enemy.lrect):
            remove_player(i)

    #Draw player to screen
    def draw(self):
        WINDOW.blit(self.surf, self.rect)

    #Draw lines from center of player to edges of gap
    def draw_lines(self):
        for enemy in enemies:
            pygame.draw.line(WINDOW, self.colour, self.rect.center, (enemy.lrect.x + WIDTH, enemy.lrect.y + enemy.HEIGHT), 2)
            pygame.draw.line(WINDOW, self.colour, self.rect.center, (enemy.rrect.x, enemy.rrect.y + enemy.HEIGHT), 2)

class Enemy:
    GAP = 200
    HEIGHT = 50
    def __init__(self):
        #Left enemy
        self.lx = random.randint(0, WIDTH - self.GAP)
        self.ly = -HEIGHT

        #Right enemy
        self.rx = self.lx + self.GAP
        self.ry = -HEIGHT

        #Enemy velocity
        self.vel = 10

        #Enemy red colour
        self.colour = (255, 0, 0)

        #Create a surface the size of an enemy
        self.surf = pygame.Surface((WIDTH, self.HEIGHT))
        self.surf.fill(self.colour)

        #Create a rect for each side enemy
        self.lrect = pygame.Rect(self.lx - WIDTH, self.ly, WIDTH, self.HEIGHT)
        self.rrect = pygame.Rect(self.rx, self.ly, WIDTH, self.HEIGHT)

    #Update enemy
    def update(self):
        global points
        #Move rects based on velocity
        self.lrect.move_ip(0, self.vel)
        self.rrect.move_ip(0, self.vel)
        #If rect goes past bottom of screen, reset enemies
        #Increment points
        if self.lrect.y > HEIGHT:
            points += 1
            self.reset()

    #Reset enemy
    def reset(self):
        #Set left side to a random x and reset to above screen
        self.lrect.x = random.randint(0, WIDTH - self.GAP)
        self.lrect.y = -self.HEIGHT

        #Set right side to GAP distance away from left enemy
        self.rrect.x = self.lrect.x + self.GAP
        self.rrect.y = -self.HEIGHT

        #Rects are drawn from top left
        self.lrect.x -= WIDTH

    #Draw enemy on screen
    def draw(self):
        WINDOW.blit(self.surf, self.lrect)
        WINDOW.blit(self.surf, self.rrect)

#Draw game to screen
def drawGame():
    #Clear window
    WINDOW.fill((255, 255, 255))

    #Draw players
    for player in players:
        player.draw()
        player.draw_lines()

    #Draw enemies
    for enemy in enemies:
        enemy.draw()

    score()
    statistics()
    
    #Update display
    pygame.display.update()

#Draw scrore onto screen
def score():
    global points
    text = FONT.render("Points:  " + str(points), True, (0, 0, 0))
    WINDOW.blit(text, (WIDTH - text.get_width(), 0))

#Draw generation statistics
def statistics():
    global ge
    text_1 = FONT.render("Players Alive: " + str(len(players)), True, (0,0,0))
    text_2 = FONT.render("Generation: " + str(population.generation+1), True, (0,0,0))

    WINDOW.blit(text_1, (0, 0))
    WINDOW.blit(text_2, (0, 20))

#Remove player, net and genomes from lists
def remove_player(i):
    ge[i].fitness -= 100
    players.pop(i)
    ge.pop(i)
    nets.pop(i)

def distance(enemy, player):
    e = enemy.rect.center
    p = player.rect.center
    dx = e[0] - p[0]
    dy = e[1] - p[1]
    return math.sqrt(dx**2 + dy**2)

def eventLoop():
    #Event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

def eval_genomes(genomes, config):
    global players, enemies, ge, nets, points

    clock = pygame.time.Clock()

    points = 0

    #Create list of player objects
    players = []
    #Create a lost for enemies
    enemies = []
    #Dictionaries with info on every player (fitness, nodes, connections...)
    ge = []
    #Stores neural net object of each player
    nets = []

    #Initialise lists with dino object, genomes and nets
    #Set all fitnesses to start at 0
    for genome_id, genome in genomes:
        players.append(Player())
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    running = True
    while running:
        eventLoop()

        #Fill up list with enemies
        if len(enemies) == 0:
            enemies.append(Enemy())

        #Check for remaining players
        if len(players) == 0:
            break

        #Player collisions
        for enemy in enemies:
            for i, player in enumerate(players):
                player.off_screen(i)
                player.collide(enemy, i)

        #Update enemies
        for enemy in enemies:
            enemy.update()


        #Player input
        for i, player in enumerate(players):
            #Get output from neural net
            output = nets[i].activate((player.rect.x + player.SIZE // 2,
                                        abs(player.rect.x - (enemies[0].lrect.x + WIDTH)),
                                        abs(player.rect.x - enemies[0].rrect.x))) #Distance between player and obstacle

            #Player movement based on net outputs
            if output[0] >= 0.5:
                player.move(0)
                ge[i].fitness += 0.1
            elif output[1] >= 0.5:
                player.move(1)
                ge[i].fitness += 0.1

        clock.tick(60)
        drawGame()

#Setup neat
def run(config_path):
    global population
    #Create a config from config file
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    
    #Create a population
    population = neat.Population(config)

    #Stat reporters 
    #Give output about networks
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    #Call run on population for 50 generations using eval_genomes fitness function
    population.run(eval_genomes, 50)

if __name__ == '__main__':
    #Get directory of config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    #Call run function passing config file
    run(config_path)
