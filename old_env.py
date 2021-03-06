import random
import math
import numpy as np

import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import DrawOptions


width = 1000
height = 700
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

class GameState:
    def __init__(self, FPS=60, clock_FPS = 0,draw_screen=True, show_sensors=True):
        # Global-ish.
        self.crashed = False

        # Physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)

        # Create the car.
        self.create_car(500, 350, 0.5)
        self.height = 700
        self.width = 1000
        # Record steps.
        self.num_steps = 0
        self.FPS = FPS
        self.clock_FPS = clock_FPS
        self.draw_screen = draw_screen
        self.show_sensors = show_sensors
        # Create walls.
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, self.height), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, self.height-1), (self.width, self.height-1), 1),
            pymunk.Segment(
                self.space.static_body,
                (self.width-1, self.height), (self.width-1, 1), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (self.width, 1), 1)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['red']
            self.space.add(s)
        #self.space.add(static)

        # Create some obstacles, semi-randomly.
        # We'll create three and they'll move around to prevent over-fitting.
        self.obstacles = []
        self.obstacles.append(self.create_obstacle(200, 350, 100))
        self.obstacles.append(self.create_obstacle(700, 200 , 100))
        self.obstacles.append(self.create_obstacle(400, 550, 60))
        self.obstacles.append(self.create_obstacle(800, 550, 60))
        self.obstacles.append(self.create_obstacle(200, 110, 60))
        self.obstacles.append(self.create_obstacle(500, 400, 60))

        # Create a cat.
        self.create_cat()
        self.create_cat2()

    def create_obstacle(self, x, y, r):
        #c_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        c_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        c_shape = pymunk.Circle(c_body, r)
        c_shape.elasticity = 1.0
        c_shape.density = 1
        c_body.position = x, y
        c_shape.color = THECOLORS["blue"]
        self.space.add(c_body, c_shape)
        return c_body

    def create_cat(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.cat_body = pymunk.Body(1, inertia)
        self.cat_body.position = 50, self.height - 100
        self.cat_shape = pymunk.Circle(self.cat_body, 30)
        self.cat_shape.color = THECOLORS["orange"]
        self.cat_shape.elasticity = 1.0
        self.cat_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.space.add(self.cat_body, self.cat_shape)

    def create_cat2(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.cat_body2 = pymunk.Body(1, inertia)
        self.cat_body2.position = 400, self.height - 500
        self.cat_shape2 = pymunk.Circle(self.cat_body2, 30)
        self.cat_shape2.color = THECOLORS["orange"]
        self.cat_shape2.elasticity = 1.0
        self.cat_shape2.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.cat_body2.angle)
        self.space.add(self.cat_body2, self.cat_shape2)

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
            self.move_cat2()

        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = 100 * driving_direction

        # Update the screen and stuff.
        screen.fill(THECOLORS["black"])
        options = pymunk.pygame_util.DrawOptions(screen)
        self.space.debug_draw(options)
        #draw(screen, self.space)
        self.space.step(1./self.FPS)
        if self.draw_screen:
            pygame.display.flip()
        
        if self.clock_FPS:
            clock.tick(self.clock_FPS)
        else:
            clock.tick()

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
            reward = int(self.sum_readings(readings) / 10) - 5
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

    def move_cat2(self):
        speed = random.randint(20, 200)
        self.cat_body2.angle -= random.randint(-1, 1)
        direction = Vec2d(1, 0).rotated(self.cat_body2.angle)
        self.cat_body2.velocity = speed * direction

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
                #draw(screen, self.space)
                options = pymunk.pygame_util.DrawOptions(screen)
                self.space.debug_draw(options)
                self.space.step(1./self.FPS)
                if self.draw_screen:
                    pygame.display.flip()
                if self.clock_FPS:
                    clock.tick(self.clock_FPS)
                else:
                    clock.tick()

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
        arm_middle = list(arm_left)
        arm_right = list(arm_left)

        # Rotate them and get readings.
        readings.append(self.get_arm_distance(arm_left, x, y, angle, 0.75))
        readings.append(self.get_arm_distance(arm_middle, x, y, angle, 0))
        readings.append(self.get_arm_distance(arm_right, x, y, angle, -0.75))

        if self.show_sensors:
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
                    or rotated_p[0] >= self.width or rotated_p[1] >= self.height:
                return i  # Sensor is off the screen.
            else:
                obs = screen.get_at(rotated_p)
                if self.get_track_or_not(obs) != 0:
                    return i

            if self.show_sensors:
                pygame.draw.circle(screen, (255, 255, 255), rotated_p, 2)

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
        # new_y = self.height - (y_change + y_1)
        new_y = y_change + y_1
        return int(new_x), int(new_y)

    def get_track_or_not(self, reading):
        if reading == THECOLORS['black']:
            return 0
        else:
            return 1
    
    def convert_coordinates_pymunk_to_pygame(self, point):
        # pymunk top,left = 0,0
        # pygame bottom,left = 0, 0
        # x is the same ie width
        # y changes, i.e. height
        return point[0], self.height - point[1]
        

if __name__ == "__main__":
    # PyGame init
    # Turn off alpha since we don't use it.
    screen.set_alpha(None)
    run = True
    if run:
        game_state = GameState(60,60,True,True)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
            game_state.frame_step((random.randint(0, 2)))