import time
from collections import deque

import gym
import numpy as np
import wandb

from Agents.DDQNAgent import DDQNAgent
from FileHandler import load_model, save_model, save_outstr
from Constants import ENVIRONMENT, MAX_EPISODE, MAX_STEP, \
    RENDER_GAME_WINDOW, TRAIN_MODEL, SAVE_MODELS, SAVE_MODEL_INTERVAL

environment = gym.make(ENVIRONMENT)  # Get env
agent = DDQNAgent(environment)  # Create Agent


def ddqn_run():
    start_episode = load_model(agent)

    last_100_ep_reward = deque(maxlen=100)  # Last 100 episode rewards # TODO: What is the deque function??
    total_step = 1  # Cumulative sum of all steps in episodes
    for episode in range(start_episode, MAX_EPISODE):

        start_time = time.time()  # Keep time
        state = environment.reset()  # Reset env

        state = agent.pre_process(state)  # Process image

        state = np.stack(
            (state, state, state, state))  # TODO: Why does this functions take 4 exactly identical objects??

        total_max_q_val = 0  # Total max q vals
        total_reward = 0  # Total reward for each episode
        total_loss = 0  # Total loss for each episode
        for step in range(MAX_STEP):

            if RENDER_GAME_WINDOW:
                environment.render()  # Show state visually

            action = agent.act(state)  # Act

            # TODO: What is next_state??
            next_state, reward, done, info = environment.step(action)  # Observe

            next_state = agent.pre_process(next_state)  # Process image

            # TODO: What is happening below??
            next_state = np.stack((next_state, state[0], state[1], state[2]))

            agent.store_results(state, action, reward, next_state, done)  # Store to mem

            state = next_state  # Update state

            if TRAIN_MODEL:
                # Perform one step of the optimization (on the target network)
                loss, max_q_val = agent.train()  # Train with random BATCH_SIZE state taken from mem
            else:
                loss, max_q_val = [0, 0]

            total_loss += loss
            total_max_q_val += max_q_val
            total_reward += reward
            total_step += 1

            # TODO: Why is epsilon only updated every 1000 steps??
            if total_step % 1000 == 0:
                agent.adaptive_epsilon()  # Decrease epsilon

            if done:  # Episode completed
                current_time = time.time()  # Keep current time
                time_passed = current_time - start_time  # Find episode duration
                current_time_format = time.strftime("%H:%M:%S", time.gmtime())  # Get current dateTime as HH:MM:SS
                epsilon_dict = {'epsilon': agent.epsilon}  # Create epsilon dict to save model as file

                if SAVE_MODELS and episode % SAVE_MODEL_INTERVAL == 0:  # Save model as file
                    save_model(agent, episode, epsilon_dict)

                if TRAIN_MODEL:
                    agent.target_model.load_state_dict(agent.online_model.state_dict())  # Update target model

                last_100_ep_reward.append(total_reward)
                avg_max_q_val = total_max_q_val / step

                out_str = "Episode:{} Time:{} Reward:{:.2f} Loss:{:.2f} Last_100_Avg_Rew:{:.3f} Avg_Max_Q:{:.3f} " \
                          "Epsilon:{:.2f} Duration:{:.2f} Step:{} TStep:{}".format(
                    episode, current_time_format, total_reward, total_loss, np.mean(last_100_ep_reward), avg_max_q_val,
                    agent.epsilon, time_passed, step, total_step
                )

                # TODO: Should we keep track of rounds were loss is 0??
                if loss != 0:
                    wandb.log(
                        {"Loss": total_loss, "Average reward": np.mean(last_100_ep_reward), "Reward": total_reward,
                         "Epsilon": agent.epsilon})

                print(out_str)

                if SAVE_MODELS:
                    save_outstr(out_str)

                break
