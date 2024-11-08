import argparse
from  utils import load_data
from model import Perceptron, VotedPerceptron, AveragePerceptron

try:
    import warnings
    warnings.filterwarnings('ignore')
except:
    pass

def main(args):
    # load data
    DIR = f'./{args.dir}/'
    data = f'{args.data}/'

    train='train.csv'
    test='test.csv'

    train_X, train_y = load_data(DIR + data+ train)
    test_X, test_y = load_data(DIR + data + test)

    Model = Perceptron
    
    if args.model.startswith('s'):
        Model = Perceptron
    elif args.model.startswith('v'):
        Model = VotedPerceptron
    elif args.model.startswith('a'):
        Model = AveragePerceptron

    if args.q:
        if args.q == '2a':
            Model = Perceptron
        elif args.q == '2b':
            Model = VotedPerceptron
        elif args.q == '2c':
            Model = AveragePerceptron

    # start train
    model = Model(learning_rate=args.r, epochs=args.t)
    model.fit(train_X,train_y)
    # Learned weight vector
    model.get_weight(verbose=True)
    # Average prediction error
    model.evaluate(test_X,test_y, verbose=True)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='../Data', help='Directory of Data folder. (Default: ../Data, ex: /path/to/Data)')
    parser.add_argument('--data', type=str, default='banknote', help='Choose Dataset: banknote. (Default: banknote)')
    parser.add_argument('-M','--model', type=str, default='standard', help='Choose model. (options: standard, voted, average)')
    parser.add_argument('-T','--t', type=int, default=10, help='Set maximum epoch. (Default: 10)')
    parser.add_argument('-R','--r', type=float, default=0.01, help='Set learning rate. (Default: 0.01)')
    parser.add_argument('-q', type=str, default='', help='Choose question number. (options: 2a, 2b, 2c)')

    args = parser.parse_args()

    if args.q:
        if args.q == "2a":
            args.model = "standard" 
            args.t = 10
        elif args.q == "2b":
            args.model = "voted"
            args.t = 10
        elif args.q == "2c":
            args.model = "average"
            args.t = 10
        else:
            print("[Warn] Wrong Question number. Replaced with Question 2a")
            args.q = "2a"
            args.t = 10
        print(f"{'-'*29} Question {args.q}{'-'*29}")
    main(args)