import numpy as np
import argparse
from gradient import batch_gradient_descent, stochastic_gradient_descent, get_optional_weight
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

    train_data = np.loadtxt(DIR + data + train, delimiter=',', skiprows=1)
    test_data = np.loadtxt(DIR + data + test, delimiter=',', skiprows=1)

    # split X, Y
    train_X = train_data[:, :-1] 
    train_Y = train_data[:, -1]  
    test_X = test_data[:, :-1]   
    test_Y = test_data[:, -1]  

    if args.q == "4c":
        W = get_optional_weight(train_X,train_Y)
        print(f"Optimal Weight: {W}")
        return

    # choose modeltswidth('s') else batch_gradient_descent
    gradient_descent = stochastic_gradient_descent if args.model.startswith('s') else batch_gradient_descent
    print(f"[Start] {gradient_descent.__name__}, data: {args.data}, r: {args.r}")
    
    W, Train_cost, Test_cost =[],[],[]

    # check learning rate range
    if args.r:
        R=[args.r]
    else:
        R=[1/2**i for i in range(20)]

    # start train
    for i, r in enumerate(R):
        weights, train_cost_history, test_cost_history = gradient_descent(train_X, train_Y, test_X, test_Y, r=r)
        W.append(weights)
        Train_cost.append(train_cost_history)
        Test_cost.append(test_cost_history)


        print("-"*85,f"[{i+1}th]")
        print(f"l-rate: {r:8.5f} \n train: {Train_cost[-1][-1]:.5f} \n  test: {Test_cost[-1][-1]:.5f} ")
        print(f"Weight: {W[-1]}")
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='Data', help='Directory of Data folder. (Default: ../Data, ex: /path/to/Data)')
    parser.add_argument('--data', type=str, default='concrete', help='Choose Dataset: car, bank. (Default: car)')
    parser.add_argument('-M','--model', type=str, default='batch', help='Choose batch or stochastic Gradient Descent. (options: batch, gradient)')
    parser.add_argument('-R','--r', type=float, help='Set learning rate. (Default: 1,...,2^20, ex: 0.01)')
    parser.add_argument('-q', type=str, default='', help='Choose question number. (options: 4a, 4b, 4c)')

    args = parser.parse_args()

    if args.q:
        if args.q == "4a":
            args.model = "batch" 
            r=""
        elif args.q == "4b":
            args.model = "stochastic"
            r=""
        else:
            print("[Warn] Wrong Question number. Replaced with Question 4a")
            args.q = "4a"
        print(f"{'-'*29} Question {args.q}{'-'*29}")
    main(args)