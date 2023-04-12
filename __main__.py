import argparse

from train import train


def main() -> None:
    print("Starting")
    parser = argparse.ArgumentParser(
        description='CBIS-DDSM Classification with Multi-Agent Reinforcement Learning')
    parser.add_argument('--mass', action="store_true",
                        help='Test Mass Dataset')
    parser.add_argument('--calc', action="store_true",
                        help='Test Calcification Dataset')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')
    parser.add_argument('--n_agents', type=int, default=5,
                        help='Number of agents')
    parser.add_argument('--steps', type=int, default=20,
                        help='Number of steps agents take per episode')
    parser.add_argument('--step_size', type=int, default=10,
                        help='Size of agents movement in pixels')
    parser.add_argument('--window_size', type=int, default=32,
                        help='Size of agent\'s view window (f, where the window is f*f pixels)')
    parser.add_argument('--belief_lstm_size', type=int,
                        default=128, help='Size of belief LSTM')
    parser.add_argument('--action_lstm_size', type=int,
                        default=128, help='Size of action LSTM')
    parser.add_argument('--state_size', type=int, default=64,
                        help='Size of the information about agent\'s state')
    parser.add_argument('--hidden_layer_size_belief', type=int, default=20,
                        help='Size of hidden layer for training agent\'s belief unit')
    parser.add_argument('--hidden_layer_size_action', type=int, default=20,
                        help='Size of hidden layer for training agent\'s action unit')
    parser.add_argument('--message_size', type=int, default=64,
                        help='Size of message exchanged between agents')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--epsilon', type=float, default=0.9995,
                        help='Epsilon for epsilon-greedy exploration')
    parser.add_argument('--epsilon_decay', type=float,
                        default=0.99, help='Decay rate for epsilon')
    parser.add_argument('--mlflow_id', type=str, default="train_cbis",
                        help='Run ID for MLFlow (used to identify different experiments)')

    args = parser.parse_args()

    mass = args.mass
    calc = args.calc

    dataset_to_train = ""
    if mass and not calc:
        dataset_to_train = "mass"
    elif calc and not mass:
        dataset_to_train = "calc"
    else:
        dataset_to_train = "mass"

    train(
        dataset_to_train,
        args.mlflow_id,
        args.n_epochs,
        args.steps,
        args.cuda,
        args.n_agents,
        args.belief_lstm_size,
        args.action_lstm_size,
        args.hidden_layer_size_belief,
        args.hidden_layer_size_action,
        args.state_size,
        args.message_size,
        args.window_size,
        args.step_size,
        args.epsilon,
        args.epsilon_decay,
        args.learning_rate,
        args.batch_size,
    )

if __name__ == "__main__":
    main()
