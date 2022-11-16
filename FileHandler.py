import os

import torch
import json

from Agents.DQNAgent import DQNAgent
from Agents.DDQNAgent import DDQNAgent
from Constants import LOAD_MODEL_FROM_FILE, MODEL_PATH, LOAD_FILE_EPISODE


def load_model(agent: DQNAgent | DDQNAgent) -> int:
    """ Loads model from file

    :param agent: the running agent
    :return: int
    """
    if LOAD_MODEL_FROM_FILE:
        start_episode, start_episode_file = get_last_episode(MODEL_PATH)
        if type(agent) == DQNAgent:
            agent.target.load_state_dict(torch.load(MODEL_PATH + str(start_episode_file) + ".pkl"))

            with open(MODEL_PATH + str(start_episode_file) + '.json') as outfile:
                param = json.load(outfile)
                agent.epsilon = param.get('epsilon')

            return start_episode
        else:
            agent.target_model.load_state_dict(torch.load(MODEL_PATH + str(start_episode_file) + ".pkl"))

            with open(MODEL_PATH + str(start_episode_file) + '.json') as outfile:
                param = json.load(outfile)
                agent.epsilon = param.get('epsilon')

            return start_episode
    else:
        start_episode = 1
        return start_episode


def save_model(agent: DQNAgent | DDQNAgent, episode: int, epsilon_dict: dict):
    """ Saves the model to file

    :param agent: the agent in the running is derived from the chosen model
    :param episode: the episode to save
    :param epsilon_dict: the epsilon_dict
    """
    weights_path = MODEL_PATH + "assault-cnn-" + str(episode) + '.pkl'
    epsilon_path = MODEL_PATH + "assault-cnn-" + str(episode) + '.json'

    if type(agent) == DQNAgent:
        torch.save(agent.target.state_dict(), weights_path)
        with open(epsilon_path, 'w') as outfile:
            json.dump(epsilon_dict, outfile)
    else:
        torch.save(agent.target_model.state_dict(), weights_path)
        with open(epsilon_path, 'w') as outfile:
            json.dump(epsilon_dict, outfile)


def save_outstr(out_str: str):
    """Saves the out_str to file

    :param out_str: the out_str to save
    """
    outputPath = MODEL_PATH + "out" + '.txt'  # Save outStr to file
    with open(outputPath, 'a') as outfile:
        outfile.write(out_str + "\n")


def get_last_episode(episodes_folder: str):
    files = os.listdir(episodes_folder)
    files_to_use = []

    # Loops through all file in folder and selects only files with .pkl extension
    for file in files:
        if '.pkl' in file:
            files_to_use.append(file.split("-"))

    # Loops through selected files and finds the one with the highest number
    last_episode_num = 0
    for file in files_to_use:
        episode_num = int(file[2].split(".")[0])
        if last_episode_num < episode_num:
            last_episode_num = episode_num

    # Loops through and finds the file with the highest number and returns it
    last_file_episode = ""
    for file in files_to_use:
        if "{}.pkl".format(last_episode_num) in file[2]:
            last_file_episode = "-".join(file).split(".")[0]
            break

    return last_episode_num, last_file_episode
