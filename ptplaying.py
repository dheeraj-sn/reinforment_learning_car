"""
Once a model is learned, use this to play it.
"""

import car_environment
import numpy as np
from ptnn import neural_net
import torch
import pygame
NUM_SENSORS = 3


def play(model):

    car_distance = 0
    game_state = car_environment.GameState()

    # Do nothing to get initial.
    _, state = game_state.frame_step((2))
    state = torch.from_numpy(state)
    # Move.
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
        car_distance += 1

        # Choose action.
        action = (torch.argmax(model(state.float())))

        # Take action.
        _, state = game_state.frame_step(action)
        state = torch.from_numpy(state)

        # Tell us something.
        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)


if __name__ == "__main__":
    saved_model = './saved-models/128-128-64-50000-10000.pt'
    model = neural_net(NUM_SENSORS, [128, 128])
    model.load_state_dict(torch.load(saved_model))
    play(model)
