from Runners.DQNRunner import dqn_run
from Runners.DDQNRunner import ddqn_run

import wandb

wandb.init(project="Assault-DQN", entity="idatt2502-assault")

wandb.config = {
    "learning_rate": 0.00025,
    "epochs": 10000,
    "batch_size": 64
}

chosen = int(input("Please select which model to run:\n1. DQN\n2. Double DQN\n"))

match chosen:
    case 1:
        print("You chose to run the DQN model.\nStarting model...\n")
        dqn_run()
        pass
    case 2:
        print("You chose to run the Double DQN model.\nStarting model...\n")
        ddqn_run()
        pass
