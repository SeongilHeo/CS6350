from math import gamma
from utils import *
from model import Primal, Dual
import argparse
try:
    import warnings
    warnings.filterwarnings('ignore')
except:
    pass

def calculate_overlap(support_vectors_list):
    overlaps = []
    for i in range(len(support_vectors_list) - 1):
        overlap = len(set(support_vectors_list[i]).intersection(support_vectors_list[i + 1]))
        overlaps.append(overlap)
    return overlaps

def main(args):
    # load data
    DIR = f'./{args.dir}/'
    data = f'{args.data}/'

    train='train.csv'
    test='test.csv'

    train_X, train_y = load_data(DIR + data + train)
    test_X, test_y = load_data(DIR + data + test)

    train_y = convert_label(train_y)
    test_y = convert_label(test_y)

    if args.q == "2a" or args.q == "2b" :
        C_values = [100/873, 500/873, 700/873]

        for C in C_values:
            print(f"Running for C: {C:.4f}")
            model = Primal(schedule=args.schedule,  
                        r=args.learning_rate, 
                        a=args.a, 
                        C=C, 
                        epochs=args.epoch)
            model.train(train_X, train_y)
            train_error = model.evaluate(train_X, train_y)
            test_error = model.evaluate(test_X, test_y)

            print(f"Train Error: {train_error} Test Error: {test_error}")
        
        return

    elif args.q =="3a":
        C_values = [100/873, 500/873, 700/873]
        for C in C_values:
            print(f"Running for C: {C:.4f}")
            model = Dual(C=C)
            model.train(train_X, train_y)
            
            print(f"Weights: {model.weights} bias: {model.bias}")

            train_error = model.evaluate(train_X, train_y)
            test_error = model.evaluate(test_X, test_y)

            print(f"Train Error: {train_error} Test Error: {test_error}")
        
        return
    
    elif args.q =="3b":
        C_values = [100/873, 500/873, 700/873]
        gamma_values = [0.1,0.5,1,5,100]
        for C in C_values:
            for gamma in gamma_values:
                print(f"Running for C: {C} gamma: {gamma}")

                model = Dual(kernel=args.kernel, C=C, gamma=gamma) 
                model.train(train_X, train_y)

                train_error = model.evaluate(train_X, train_y)
                test_error = model.evaluate(test_X, test_y)

                print(f"Train Error: {train_error} Test Error: {test_error}")
        
        return
    
    elif args.q =="3c":
        C_values = [100/873, 500/873, 700/873]
        gamma_values = [0.01, 0.1,0.5,1,5]

        # Store results
        results = []
        support_vectors_all = {}

        for C in C_values:
            support_vectors_list = []
            for gamma in gamma_values:
                print(f"Running for C: {C} gamma: {gamma}")

                model = Dual(kernel=args.kernel, C=C, gamma=gamma) 
                model.train(train_X, train_y)
                
                support_indices = model.support_indices
                num_support_vectors = len(support_indices)
                
                results.append((C, gamma, num_support_vectors))
                support_vectors_list.append(support_indices)
                print(f"Gamma = {gamma:.4f}, Support Vectors = {num_support_vectors}")

            if C == 500 / 873:
                support_vectors_all[C] = support_vectors_list

        print("\nCalculating overlaps for C = 500 / 873")
        overlaps = calculate_overlap(support_vectors_all[500 / 873])
        for i in range(len(overlaps)):
            print(f"Overlap between gamma = {gamma_values[i]} and gamma = {gamma_values[i + 1]}: {overlaps[i]}")

        return

    else:
        if args.model == "primal":
            model = Primal(schedule=args.schedule, 
                        r=args.learning_rate, 
                        a=args.a, 
                        C=args.c, 
                        epochs=args.epoch)
        else:
            model = Dual(kernel=args.kernel,
                        C=args.c, 
                        gamma=args.gamma)


        model.train(train_X, train_y)
        train_error = model.evaluate(train_X, train_y)
        test_error = model.evaluate(test_X, test_y)

        print(f"Model: {args.model}")

        if args.model == "primal":
            if args.schedule: print(f"Schedule: type {args.schedule}")
            if args.learning_rate: print(f"Learning Rate: {args.learning_rate}")
            if args.a: print(f"Hyperparameter A: {args.a}")

        elif args.model == "dual":
            if args.kernel: print(f"Kernel {args.kernel}")
            if args.gamma: print(f"gamma: {args.gamma}")

        if args.c: print(f"Hyperparameter C: {args.c}")
        if args.epoch: print(f"Epoch: {args.epoch}")
        print("="*50)
        print(f"Train Error: {train_error} Test Error: {test_error}")


        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='../Data', help='Directory of Data folder. (Default: ../Data, ex: /path/to/Data)')
    parser.add_argument('--data', type=str, default='banknote', help='Choose Dataset: banknote. (Default: banknote)')
    parser.add_argument('-M','--model', type=str, default='primal', help='Choose SVM model. (options: primal, dual)')
    parser.add_argument('-K','--kernel', type=str, default=None, help='Choose kernel mode. (options: guassian)')
    parser.add_argument('-S','--schedule', type=str, default=None, help='Choose schedule type. (options: A, B)')
    parser.add_argument('-R','--learning_rate', type=float, default=0.1, help='Set learning rate. (Default: 0.1)')
    parser.add_argument('-A','--a', type=float, default=1, help='Set hyperparameter of scedule. (Default: 1)')
    parser.add_argument('-C','--c', type=float, default=100/873, help='Set hyperparameter of model. (Default: 100/873)')
    parser.add_argument('-E','--epoch', type=int, default=100, help='Set epoch. (Default: 100)')
    parser.add_argument('-G','--gamma', type=float, default=1, help='Set gamma. (Default: 1)')
    parser.add_argument('-q', type=str, default=None, help='Choose question number. (options: 2a, 2b, 3a, 3b, 3c)')

    args = parser.parse_args()

    if args.q:
        if args.q == "2a":
            args.model = "primal"
            args.learning_rate = 0.1
            args.epoch = 100
            args.schedule = "A"
            args.a = 1
        elif args.q == "2b":
            args.model = "primal"
            args.learning_rate = 0.1
            args.epoch = 100
            args.schedule = "B"
            args.a = 1
        elif args.q == "3a":
            args.model = "dual"
            args.kernel = None
        elif args.q == "3b":
            args.model = "dual"
            args.kernel = "gaussian"
        elif args.q == "3c":
            args.model = "dual"
            args.kernel = "gaussian"
        else:
            print("[Warn] Wrong Question number. Replaced with Question 2a")
            args.q = "2a"
            args.model = "primal"
            args.learning_rate = 0.1
            args.epoch = 100
            args.schedule = "A"
            args.a = 1
        print(f"{'-'*29} Question {args.q}{'-'*29}")
    main(args)