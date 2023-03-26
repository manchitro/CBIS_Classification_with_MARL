import argparse

from dataset import CBISDataset

parser = argparse.ArgumentParser(description='CBIS-DDSM Classification with Multi-Agent Reinforcement Learning')
parser.add_argument('--n_agents', type=int, default=5, help='Number of agents')
parser.add_argument('--steps', type=int, default=100, help='Number of steps agents take per episode')
parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
parser.add_argument('--img_size', type=int, default=32, help='Size of image (x, where the image is x*x pixels)')
parser.add_argument('--window_size', type=int, default=5, help='Size of agent\'s view window (f, where the window is f*f pixels)')
parser.add_argument('--hidden_belief', type=int, default=128, help='Size of belief LSTM')
parser.add_argument('--hidden_action', type=int, default=128, help='Size of action LSTM')
parser.add_argument('--state_size', type=int, default=8, help='Size of the information about agent\'s state')
parser.add_argument('--message_hidden_layer_size', type=int, default=64, help='Size of hidden layer for training agent\'s messages')
parser.add_argument('--message_size', type=int, default=16, help='Size of message exchanged between agents')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--epsilon', type=float, default=0.5, help='Epsilon for epsilon-greedy exploration')
parser.add_argument('--epsilon_decay', type=float, default=0.99, help='Decay rate for epsilon')
parser.add_argument('--mlflow_id', type=str, default="train_cbis", help='Run ID for MLFlow (used to identify different experiments)')

args = parser.parse_args()

dataset_constructor = CBISDataset

dataset = dataset_constructor("cbis")

dataset.__getitem__(0)