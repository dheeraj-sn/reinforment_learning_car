# import car_environment
import old_env
import numpy as np
from pt_convdqn import neural_net
import torch
import pygame
NUM_SENSORS = 3


def play(model):

    car_distance = 0
    game_state = old_env.GameState(60,60,True,True)

    # Do nothing to get initial.
    _, state = game_state.frame_step((2))
    state = torch.from_numpy(state)
    
    model.eval()
    
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
    saved_model = './saved-models/CNN - 512-512-32-50000-300000.pt'
    model = neural_net(NUM_SENSORS, [512, 512])
    model.load_state_dict(torch.load(saved_model))
    play(model)
