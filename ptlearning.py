import old_env
import numpy as np
import random
import csv
from ptnn import neural_net
import os.path
import timeit
import torch

NUM_INPUT = 3
GAMMA = 0.9  # Forgetting.
TUNING = False  # If False, just use arbitrary, pre-selected params.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device : ",device)

def train_net(model, params, mseloss, optimizer):

    filename = params_to_filename(params)

    observe = 1000  # Number of frames to observe before training.
    epsilon = 1
    train_frames = 100000  # Number of frames to play.
    batchSize = params['batchSize']
    buffer = params['buffer']

    # Just stuff used below.
    max_car_distance = 0
    car_distance = 0
    t = 0
    data_collect = []
    replay = []  # stores tuples of (S, A, R, S').

    loss_log = []

    game_state = old_env.GameState(FPS=10, clock_FPS=0 ,draw_screen = False, show_sensors = False)
    # Get initial state by doing nothing and getting the state.
    _, state = game_state.frame_step((2))
    state = torch.from_numpy(state)
    state = state.to(device)
    
    # Let's time it.
    start_time = timeit.default_timer()
    
    # Run the frames.
    while t < train_frames:

        t += 1
        car_distance += 1

        # Choose an action.
        if random.random() < epsilon or t < observe:
            action = np.random.randint(0, 3)  # random
        else:
            # Get Q values for each action.
            qval = model(state.float())
            action = (torch.argmax(qval))  # best value

        # Take action, observe new state and get our treat.
        reward, new_state = game_state.frame_step(action)
        new_state = torch.from_numpy(new_state)
        new_state = new_state.to(device)
        # Experience replay storage.
        replay.append((state, action, reward, new_state))

        # If we're done observing, start training.
        if t > observe:

            # If we've stored enough in our buffer, pop the oldest.
            if len(replay) > buffer:
                replay.pop(0)

            # Randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)
            
            optimizer.zero_grad()
            
            # Get training values.
            X_train, y_train = process_minibatch2(minibatch, model)
            # Train the model on this batch.
            X_train, y_train = X_train.to(device), y_train.to(device)
            X_train.requires_grad_()
            score = model(X_train)
            loss = mseloss(score,y_train)
            loss.backward()
            optimizer.step()
            loss_log.append(loss.detach().item())

        # Update the starting state with S'.
        state = new_state

        # Decrement epsilon over time.
        if epsilon > 0.1 and t > observe:
            epsilon -= (1.0/train_frames)

        # We died, so update stuff.
        if reward == -500:
            # Log the car's distance at this T.
            data_collect.append([t, car_distance])

            # Update max.
            if car_distance > max_car_distance:
                max_car_distance = car_distance

            # Time it.
            tot_time = timeit.default_timer() - start_time
            fps = car_distance / tot_time

            # Output some stuff so we can watch.
            print("Max: %d at %d\tepsilon %f\t(%d)\t%f fps" %
                  (max_car_distance, t, epsilon, car_distance, fps))

            # Reset.
            car_distance = 0
            start_time = timeit.default_timer()

        # Save the model every 25,000 frames.
        if t % 25000 == 0:
            torch.save(model.state_dict(),'saved-models/' + filename + '-' +
                               str(t) + '.pt')
            print("Saving model %s - %d" % (filename, t))
    print(data_collect)
#     print(loss_log)
    
    # Log results after we're done all frames.
    log_results(filename, data_collect, loss_log)


def log_results(filename, data_collect, loss_log):
    # Save the results to a file so we can graph it later.
    with open('results/new-sonar-frames/learn_data-' + filename + '.csv', 'w') as data_dump:
        wr = csv.writer(data_dump)
        wr.writerows(data_collect)

    with open('results/new-sonar-frames/loss_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow([loss_item])

def process_minibatch2(minibatch, model):
    # by Microos, improve this batch processing function 
    #   and gain 50~60x faster speed (tested on GTX 1080)
    #   significantly increase the training FPS
    
    # instead of feeding data to the model one by one, 
    #   feed the whole batch is much more efficient

    mb_len = len(minibatch)

    old_states = torch.zeros((mb_len, 3))
    actions = torch.zeros((mb_len,))
    rewards = torch.zeros((mb_len,))
    new_states = torch.zeros((mb_len, 3))

    old_states = old_states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    new_states = new_states.to(device)


    for i, m in enumerate(minibatch):
        old_state_m, action_m, reward_m, new_state_m = m
        old_states[i, :] = old_state_m[...]
        actions[i] = action_m
        rewards[i] = reward_m
        new_states[i, :] = new_state_m[...]
    
    
    old_qvals = model(old_states)
    old_qvals = old_qvals.to(device)
    new_qvals = model(new_states)
    new_qvals = new_qvals.to(device)

    maxQs = torch.max(new_qvals,1)
    y = old_qvals
    non_term_inds = torch.where(rewards != -500)[0]
    term_inds = torch.where(rewards == -500)[0]
    y[non_term_inds, actions[non_term_inds].long()] = rewards[non_term_inds] + (GAMMA * maxQs.values[non_term_inds])
    y[term_inds, actions[term_inds].long()] = rewards[term_inds]
    X_train = old_states
    y_train = y
    return X_train, y_train

def process_minibatch(minibatch, model):
    """This does the heavy lifting, aka, the training. It's super jacked."""
    X_train = []
    y_train = []
    # Loop through our batch and create arrays for X and y
    # so that we can fit our model at every step.
    for memory in minibatch:
        # Get stored values.
        old_state_m, action_m, reward_m, new_state_m = memory
        # Get prediction on old state.
        old_qval = model.predict(old_state_m, batch_size=1)
        # Get prediction on new state.
        newQ = model.predict(new_state_m, batch_size=1)
        # Get our predicted best move.
        maxQ = np.max(newQ)
        y = np.zeros((1, 3))
        y[:] = old_qval[:]
        # Check for terminal state.
        if reward_m != -500:  # non-terminal state
            update = (reward_m + (GAMMA * maxQ))
        else:  # terminal state
            update = reward_m
        # Update the value for the action we took.
        y[0][action_m] = update
        X_train.append(old_state_m.reshape(NUM_INPUT,))
        y_train.append(y.reshape(3,))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train


def params_to_filename(params):
    if(len(params['nn'])<=2):
        return str(params['nn'][0]) + '-' + str(params['nn'][1]) + '-' + \
            str(params['batchSize']) + '-' + str(params['buffer'])
    else:
        return 'LSTM - ' + str(params['nn'][0]) + '-' + str(params['nn'][1]) + '-' + str(params['nn'][2]) + '-' + \
            str(params['batchSize']) + '-' + str(params['buffer'])


def launch_learn(params):
    filename = params_to_filename(params)
    print("Trying %s" % filename)
    # Make sure we haven't run this one.
    if not os.path.isfile('results/new-sonar-frames/loss_data-' + filename + '.csv'):
        # Create file so we don't double test when we run multiple
        # instances of the script at the same time.
        open('results/new-sonar-frames/loss_data-' + filename + '.csv', 'a').close()
        print("Starting test.")
        # Train.

        model = neural_net(NUM_INPUT, params['nn'])
        model = model.to(device)
        train_net(model, params)
    else:
        print("Already tested.")


if __name__ == "__main__":
    if TUNING:
        param_list = []
        nn_params = [[164, 150], [256, 256],
                     [512, 512], [1000, 1000]]
        batchSizes = [40, 100, 400]
        buffers = [10000, 50000]

        for nn_param in nn_params:
            for batchSize in batchSizes:
                for buffer in buffers:
                    params = {
                        "batchSize": batchSize,
                        "buffer": buffer,
                        "nn": nn_param
                    }
                    param_list.append(params)

        for param_set in param_list:
            launch_learn(param_set)

    else:
        nn_param = [256, 512, 512]
        params = {
            "batchSize": 32,
            "buffer": 500,
            "nn": nn_param
        }
        model = neural_net(NUM_INPUT, nn_param)
        model = model.to(device)
        optimizer = torch.optim.RMSprop(model.parameters(),lr=0.01)
        mseloss = torch.nn.MSELoss()
        train_net(model, params, mseloss, optimizer)
