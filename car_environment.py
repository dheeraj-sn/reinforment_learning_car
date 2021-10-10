import random
import math
import numpy as np

import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import DrawOptions as draw

# PyGame init
width = 1000
height = 700
left = 0
right = width -1
top = 0
bottom = height - 1


FPS = 10
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Turn off alpha since we don't use it.
screen.set_alpha(None)

# Showing sensors and redrawing slows things down.
show_sensors = True
draw_screen = True

# Colors
color_red = pygame.Color(255, 0, 0)
color_green = pygame.Color(0, 255, 0)
color_blue = pygame.Color(0, 0, 255)
color_grey = pygame.Color(128, 128, 128)
color_orange = pygame.Color(255, 165, 0)
color_black = pygame.Color(255, 255, 255)
color_white = pygame.Color(0,0,0)

def convert_coordinates_pymunk_to_pygame(point):
    return point[0], height - point[1]
    # pymunk top,left = 0,0
    # pygame bottom,left = 0, 0
    # x is the same ie width
    # y changes, i.e. height


class Wall():
    def __init__(self,p1,p2):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.shape = pymunk.Segment(body=self.body, a=p1, b=p2, radius=50)
        self.shape.elasticity = 1
        self.shape.collision_type = 1
        self.shape.friction = 1
        self.shape.color = THECOLORS['red']
        self.shape.group = 1
        #space.add(self.body,self.shape)
    
    def draw(self):
        pygame.draw.line(surface=screen, color=self.shape.color,\
                start_pos=convert_coordinates_pymunk_to_pygame(self.shape.a), \
                end_pos=convert_coordinates_pymunk_to_pygame(self.shape.b))

class Obstacle():
    def __init__(self, x, y, r):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.shape = pymunk.Circle(body=self.body, radius=r)
        self.body.position = x, y
        self.shape.density = 1
        self.shape.elasticity = 1.0
        self.shape.color = color_blue
        #self.shape.collision_type = 1
        #space.add(self.body, self.shape)
    
    def draw(self):
        pygame.draw.circle(surface=screen, color=self.shape.color, \
            center=convert_coordinates_pymunk_to_pygame(self.body.position), \
            radius=self.shape.radius)
    
    def reset(self, space, arbiter, data):
        self.body.position = middlex, middley
        self.body.velocity = 400, -300
        return

class Cat():
    def __init__(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.body = pymunk.Body(1, inertia)
        self.body.position = 50, height - 100
        self.shape = pymunk.Circle(self.body, 30)
        self.shape.color = color_orange
        self.shape.elasticity = 1
        self.shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.body.angle)
    
    def draw(self):
        pygame.draw.circle(surface=screen, color=self.shape.color, \
            center=convert_coordinates_pymunk_to_pygame(self.body.position), \
            radius=self.shape.radius)

class Car():
    def __init__(self, x, y, r):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.body = pymunk.Body(1, inertia)
        self.body.position = x, y
        self.shape = pymunk.Circle(self.body, 25)
        self.shape.color = color_green
        self.shape.elasticity = 1.0
        self.body.angle = r
        driving_direction = Vec2d(1, 0).rotated(self.body.angle)
        self.body.apply_force_at_local_point(driving_direction)
    
    def draw(self):
        pygame.draw.circle(surface=screen, color=self.shape.color, \
            center=convert_coordinates_pymunk_to_pygame(self.body.position), \
            radius=self.shape.radius)

class GameState:
    def __init__(self):
        # Global-ish.
        self.crashed = False

        # Physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)

        # Record steps.
        self.num_steps = 0

        # Create walls.
        
        wall_left = Wall([left, top], [left, bottom])
        wall_right = Wall([right, top], [right, bottom])
        wall_top = Wall([left, top], [right, top])
        wall_bottom = Wall([left, bottom], [right, bottom])
        self.walls = [wall_left, wall_right, wall_top, wall_bottom]
        for wall in self.walls:
            self.space.add(wall.body, wall.shape)
    
        # Create some obstacles, semi-randomly.
        # We'll create three and they'll move around to prevent over-fitting.
        self.obstacles = []
        self.obstacles.append(Obstacle(200, 350, 100))
        self.obstacles.append(Obstacle(700, 200, 125))
        self.obstacles.append(Obstacle(600, 600, 35))

        # Create a cat.
        self.cat = Cat()
        self.cat_body = self.cat.body
        self.space.add(self.cat.body, self.cat.shape)
        #self.create_cat()

        # Create car
        self.car = Car(100, 100, 0.5)
        self.car_body = self.car.body
        self.space.add(self.car.body, self.car.shape)

    def draw_all(self):
        for wall in self.walls:
            wall.draw()
        for obstacle in self.obstacles:
            obstacle.draw()
        self.cat.draw()
        self.car.draw()

    """
    def create_obstacle(self, x, y, r):
        c_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        c_shape = pymunk.Circle(c_body, r)
        c_shape.elasticity = 1.0
        c_body.position = x, y
        c_shape.color = THECOLORS["blue"]
        self.space.add(c_body, c_shape)
        return c_body

    def create_cat(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.cat_body = pymunk.Body(1, inertia)
        self.cat_body.position = 50, height - 100
        self.cat_shape = pymunk.Circle(self.cat_body, 30)
        self.cat_shape.color = THECOLORS["orange"]
        self.cat_shape.elasticity = 1.0
        self.cat_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.space.add(self.cat_body, self.cat_shape)

    def create_car(self, x, y, r):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.car_body = pymunk.Body(1, inertia)
        self.car_body.position = x, y
        self.car_shape = pymunk.Circle(self.car_body, 25)
        self.car_shape.color = THECOLORS["green"]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = r
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse_at_local_point(driving_direction)
        self.space.add(self.car_body, self.car_shape)
    """
    def frame_step(self, action):
        if action == 0:  # Turn left.
            self.car_body.angle -= .2
        elif action == 1:  # Turn right.
            self.car_body.angle += .2

        # Move obstacles.
        if self.num_steps % 100 == 0:
            self.move_obstacles()

        # Move cat.
        if self.num_steps % 5 == 0:
            self.move_cat()

        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = 100 * driving_direction

        # Update the screen and stuff.
        screen.fill(color_white)

        #draw(screen)
        self.draw_all()
    
        if draw_screen:
            #pygame.display.update()
            pygame.display.flip()
        clock.tick(FPS)
        self.space.step(1./FPS)
        
        
        # Get the current location and the readings there.
        x, y = self.car_body.position
        readings = self.get_sonar_readings(x, y, self.car_body.angle)
        normalized_readings = [(x-20.0)/20.0 for x in readings] 
        state = np.array([normalized_readings])

        # Set the reward.
        # Car crashed when any reading == 1
        if self.car_is_crashed(readings):
            self.crashed = True
            reward = -500
            self.recover_from_crash(driving_direction)
        else:
            # Higher readings are better, so return the sum.
            reward = -5 + int(self.sum_readings(readings) / 10)
        self.num_steps += 1

        return reward, state
        

    def move_obstacles(self):
        # Randomly move obstacles around.
        for obstacle in self.obstacles:
            speed = random.randint(1, 5)
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            obstacle.velocity = speed * direction

    def move_cat(self):
        speed = random.randint(20, 200)
        self.cat_body.angle -= random.randint(-1, 1)
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.cat_body.velocity = speed * direction

    def car_is_crashed(self, readings):
        if readings[0] == 1 or readings[1] == 1 or readings[2] == 1:
            return True
        else:
            return False

    def recover_from_crash(self, driving_direction):
        """
        We hit something, so recover.
        """
        while self.crashed:
            # Go backwards.
            self.car_body.velocity = -100 * driving_direction
            self.crashed = False
            for i in range(10):
                self.car_body.angle += .2  # Turn a little.
                screen.fill(THECOLORS["grey7"])  # Red is scary!
                #draw(screen)
                self.draw_all()
                self.space.step(1./FPS)
                if draw_screen:
                    pygame.display.flip()
                clock.tick(FPS)

    def sum_readings(self, readings):
        """Sum the number of non-zero readings."""
        tot = 0
        for i in readings:
            tot += i
        return tot

    def get_sonar_readings(self, x, y, angle):
        readings = []
        """
        Instead of using a grid of boolean(ish) sensors, sonar readings
        simply return N "distance" readings, one for each sonar
        we're simulating. The distance is a count of the first non-zero
        reading starting at the object. For instance, if the fifth sensor
        in a sonar "arm" is non-zero, then that arm returns a distance of 5.
        """
        # Make our arms.
        arm_left = self.make_sonar_arm(x, y)
        arm_middle = arm_left
        arm_right = arm_left

        # Rotate them and get readings.
        readings.append(self.get_arm_distance(arm_left, x, y, angle, 0.75))
        readings.append(self.get_arm_distance(arm_middle, x, y, angle, 0))
        readings.append(self.get_arm_distance(arm_right, x, y, angle, -0.75))

        if show_sensors:
            pygame.display.update()

        return readings

    def get_arm_distance(self, arm, x, y, angle, offset):
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                return i  # Sensor is off the screen.
            else:
                obs = screen.get_at(rotated_p)
                if self.get_track_or_not(obs) != 0:
                    return i

            if show_sensors:
                pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)

        # Return the distance for the arm.
        return i

    def make_sonar_arm(self, x, y):
        spread = 10  # Default spread.
        distance = 20  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 40):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points

    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)
        return int(new_x), int(new_y)

    def get_track_or_not(self, reading):
        if reading == THECOLORS['black']:
            return 0
        else:
            return 1

if __name__ == "__main__":

    def run_game():
        game_state = GameState()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            game_state.frame_step((random.randint(0, 2)))
    
    run_game()
    pygame.quit()