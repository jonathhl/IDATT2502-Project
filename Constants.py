import torch

ENVIRONMENT = "ALE/Assault-v5"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_MODELS = True  # Save models to file so you can test later
MODEL_PATH = "./models/assault-cnn-"  # Models path for saving or loading
SAVE_MODEL_INTERVAL = 10  # Save models at every X epoch
TRAIN_MODEL = True  # Train model while playing (Make it False when testing a model)

LOAD_MODEL_FROM_FILE = False  # Load model from file
LOAD_FILE_EPISODE = 0  # Load Xth episode from file

BATCH_SIZE = 64  # Minibatch size that select randomly from mem for train nets
MAX_EPISODE = 100_000  # Max episode
MAX_STEP = 100_000  # Max step size for one episode

# TODO: What are the below constants used for??
MAX_MEMORY_LEN = 50_000  # Max memory len
MIN_MEMORY_LEN = 40_000  # Min memory len before start train

GAMMA = .97  # Discount rate
LEARNING_RATE = .00025  # Learning rate
EPSILON_DECAY = .99  # Epsilon decay rate by step
EPSILON = 1
EPSILON_MIN = .05

RENDER_GAME_WINDOW = False
