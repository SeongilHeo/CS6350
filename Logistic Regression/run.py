from ast import arg
from utils import *
from model import LogisticRegressionSGD
import matplotlib.pyplot as plt
import argparse
try:
    import warnings
    warnings.filterwarnings('ignore')
except:
    pass

# Visualization function
def plot_losses(models, variances):
    plt.figure(figsize=(12, 8))
    for v, model in zip(variances, models):
        plt.plot(model.train_losses, label=f'Prior Variance {v}')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main(args):
    DIR = f'./{args.dir}/'
    data = f'{args.data}/'

    train='train.csv'
    test='test.csv'

    train_X, train_y = load_data(DIR + data + train)
    test_X, test_y = load_data(DIR + data + test)

    if args.q == "3a":
        models = []
        # variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
        variances = [0.1, 0.5, 1, 3, 5, 10, 100]
        for v in variances:
            print(f"\nTesting with prior_variance={v}")
            model_map = LogisticRegressionSGD(learning_rate=0.1, max_epochs=100, prior_variance=v)
            model_map.train(train_X, train_y)
            model_map.evaluate(train_X, train_y, "train")
            model_map.evaluate(test_X, test_y, "test")
            models.append(model_map)

        plot_losses(models, variances)

    elif args.q == "3b":

        print("\nTesting with Maximum Likelihood (ML) Estimation")
        model_ml = LogisticRegressionSGD(learning_rate=0.1, max_epochs=100)
        model_ml.train(train_X, train_y)
        model_ml.evaluate(train_X, train_y, "train")
        model_ml.evaluate(test_X, test_y, "test")
        plt.figure(figsize=(12, 8))
        plt.plot(model_ml.train_losses, label=f'ML Estimation')
        plt.title('Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
                
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='../Data', help='Directory of Data folder. (Default: ../Data, ex: /path/to/Data)')
    parser.add_argument('--data', type=str, default='banknote', help='Choose Dataset: banknote. (Default: banknote)')
    parser.add_argument('-q', type=str, default=None, help='Choose question number. (options: 3a, 3b)')

    args = parser.parse_args()

    if not args.q:
        args.q = "3a"
    print(f"{'-'*29} Question {args.q}{'-'*29}")
    main(args)