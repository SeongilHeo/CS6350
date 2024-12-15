from utils import *
from model import ThreeLayerNN
from model_torch import ThreeLayerNNTorch
import matplotlib.pyplot as plt
import argparse
try:
    import warnings
    warnings.filterwarnings('ignore')
except:
    pass


def main(args):
    # load data
    DIR = f'./{args.dir}/'
    data = f'{args.data}/'


    if args.q == "2a":
        train_X, train_y = np.array([[1,1,1]]), np.array([1])

        custom_weights = {
            'W1': [[-1., 1.], [-2., 2.], [-3., 3.]],
            'W2': [[-1., 1.], [-2., 2.], [-3., 3.]],
            'W3': [[-1.],[2.],[-1.5]],
        }

        nn = ThreeLayerNN(input_size=train_X.shape[1], hidden_size1=3, hidden_size2=3, output_size=1, custom_weights=custom_weights, printdw=True)
        nn.train(train_X, train_y, epochs=1)

        return

    elif args.q == "2b":
        train='train.csv'
        test='test.csv'

        train_X, train_y = load_data(DIR + data + train)
        train_y = train_y.reshape(-1, 1)

        test_X, test_y = load_data(DIR + data + test)
        test_y = test_y.reshape(-1, 1)
        
        # Hyperparameters
        gamma_0 = 0.01
        d = 100
        widths = [5, 10, 25, 50, 100]
        epochs = 100
        for width in widths:
            print(f"Training with width={width}")
            nn = ThreeLayerNN(input_size=train_X.shape[1], hidden_size1=width, hidden_size2=width, output_size=1, gamma_0=gamma_0, d=d, stochastic=True)
            losses = nn.train(train_X, train_y, epochs)

            nn.evaluate(train_X, train_y, label="Train")
            nn.evaluate(test_X, test_y, label="Test")

            # Plot loss curve
            plt.plot(losses, label=f'Width={width}')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss vs. Epochs')
        plt.legend()
        plt.show()

        return

    elif args.q == "2c":

        train='train.csv'
        test='test.csv'

        train_X, train_y = load_data(DIR + data + train)
        train_y = train_y.reshape(-1, 1)

        test_X, test_y = load_data(DIR + data + test)
        test_y = test_y.reshape(-1, 1)

        # Hyperparameters
        gamma_0 = 0.01
        d = 100
        widths = [5, 10, 25, 50, 100]
        epochs = 100
        custom_weights="zero"
        for width in widths:
            print(f"Training with width={width}")
            nn = ThreeLayerNN(input_size=train_X.shape[1], hidden_size1=width, hidden_size2=width, output_size=1, gamma_0=gamma_0, d=d, stochastic=True, custom_weights=custom_weights)
            losses = nn.train(train_X, train_y, epochs)

            nn.evaluate(train_X, train_y, label="Train")
            nn.evaluate(test_X, test_y, label="Test")

            # Plot loss curve
            plt.plot(losses, label=f'Width={width}')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss vs. Epochs')
        plt.legend()
        plt.show()

        return
    
    elif args.q == "2e":
        train='train.csv'
        test='test.csv'

        train_X, train_y = load_data(DIR + data + train)
        test_X, test_y = load_data(DIR + data + test)
        ThreeLayerNNTorch(train_X,train_y,test_X,test_y)

            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='../Data', help='Directory of Data folder. (Default: ../Data, ex: /path/to/Data)')
    parser.add_argument('--data', type=str, default='banknote', help='Choose Dataset: banknote. (Default: banknote)')
    parser.add_argument('-q', type=str, default=None, help='Choose question number. (options: 2a, 2b, 2c, 2e)')

    args = parser.parse_args()

    if not args.q:
        args.q = "2a"
    print(f"{'-'*29} Question {args.q}{'-'*29}")
    main(args)