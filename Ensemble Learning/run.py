from utils import *
from Ensemble import AdaBoost, BaggedTrees, RandomForest
import argparse

def Q2ce(Model, criterion, train_X, train_Y, labels, attributes, columns, numerical_attributes):
    print(f"Model: {Model.__name__}")
    # train model
    TREES=[]
    t=500
    for i in range(100):
        print(f"[{i+1:2}th]{'-'*64}")
        model = Model(num_trees=t, criterion=criterion, sample_ratio=1/5, repalce = False)

        model.train(train_X,
                    train_Y,
                    labels,
                    attributes,
                    columns,
                    numerical_attributes)
        TREES.append(model)

    m = len(train_Y)
    train_Y = covert_labels(train_Y)
    
    # Single Tree
    zero_Y_hat = []
    # predict data
    for T in TREES:
        Y_hat = T.trees[0].predict(train_X)
        Y_hat = covert_labels(Y_hat)
        zero_Y_hat.append(Y_hat)

    # calculate sample bias and var
    Y_hat_avg = [0] * m
    Bias = [0] * m
    Var = [0] * m
    
    for i in range(m):
        Y_hat_avg[i] = calculate_average([Y_hat[i] for Y_hat in zero_Y_hat])
        Bias[i] =(Y_hat_avg[i]-train_Y[i])**2
        Var[i] = calculate_average([(Y_hat[i]-Y_hat_avg[i])**2 for Y_hat in zero_Y_hat])

    # calcuate error
    avg_bias = calculate_average(Bias)
    avg_var = calculate_average(Var)
    GSE = avg_bias + avg_var
    
    # Bagged or RandomForest tree
    full_Y_hat = []
    # predict data
    for T in TREES:
        Y_hat = T.predict(train_X)
        Y_hat = covert_labels(Y_hat)
        full_Y_hat.append(Y_hat)

    # calculate bias and var
    full_Y_hat_avg = [0] * m
    full_Bias = [0] * m
    full_Var = [0] * m

    for i in range(m):
        full_Y_hat_avg[i] = calculate_average([Y_hat[i] for Y_hat in full_Y_hat])
        full_Bias[i] =(full_Y_hat_avg[i]-train_Y[i])**2
        full_Var[i] = calculate_average([(Y_hat[i]-full_Y_hat_avg[i])**2 for Y_hat in full_Y_hat])

    # calcuate error
    full_avg_bias = calculate_average(full_Bias)
    full_avg_var = calculate_average(full_Var)
    full_GSE = full_avg_bias + full_avg_var

    print(f"{'-'*32}[RESULT]{'-'*32}")
    print(f"Single Average Bias: {avg_bias} \t Single Average Variance: {full_avg_bias}")
    print(f"Bagged Average Bias: {avg_var} \t Bagged Average Variance: {full_avg_var}")
    print(f"Bagged Squared Error: {GSE} \t Bagged Squared Error: {full_GSE}")

    # visualize
    try:
        import numpy
        import matplotlib.pyplot
        from visualize import draw_bias_var
        draw_bias_var(Bias,full_Bias, Var, full_Var)
    except:
        pass

def Q2d(criterion, train_X, train_Y, labels, attributes, columns, numerical_attributes,test_X,test_Y):
    print(f"Model: RandomForest")
    T = 500
    result = [[0,0,0] for _ in range(T)]
    num_attributes = [2,4,6]
    # train model
    for t in range(1,T+1):
        print(f"[{t}th]{'-'*65}")
        for idx, num_attr in enumerate(num_attributes):
            print(f"------[#attr: {num_attr}]{'-'*54}")
            model = RandomForest(num_trees=T, criterion=criterion, sample_ratio=args.ratio,  num_attributes=num_attr)
            model.train(train_X,
                    train_Y,
                    labels,
                    attributes,
                    columns,
                    numerical_attributes)
            
            # evaluate train_Y
            print("[TRAIN]",end=" ")
            train_err = 1 - model.evaluate(train_X,train_Y,verbose=True)
            # evaluate test_Y
            print("[TEST] ",end=" ")
            test_err = 1 - model.evaluate(test_X, test_Y, verbose=True)

            result[t-1][idx] = [train_err,test_err]
    try:
        import numpy
        import matplotlib.pyplot
        from visualize import draw_randomforest
        draw_randomforest(result)
    except:
        pass


def main(args):
    DIR = f'./{args.dir}/{args.data}/'
    file_train = "train.csv"
    file_test = "test.csv"
    file_desc = "data-desc.txt"

    # load train csv
    csv_train=DIR+file_train
    train_X, train_Y = load_data(csv_train)
    
    # load test csv
    dotest = True
    try:
        csv_test=DIR+file_test
        test_X, test_Y = load_data(csv_test)
    except:
        dotest=False
        pass

    # load data desc
    labels, attributes, categorical_attributes, numerical_attributes, missing_attributes, columns = load_data_desc(DIR+file_desc)

    if numerical_attributes:
        train_X = preprosess_numeric(train_X, numerical_attributes, columns)
        if dotest:
            test_X = preprosess_numeric(test_X, numerical_attributes, columns)

    criterion = 'information_gain'

    # Selected Question 
    if args.q:    
        print(f"{'-'*29} Question {args.q}{'-'*29}")
    if args.q == '2c' or args.q == '2e':
        Model = BaggedTrees if args.q == '2c' else RandomForest
        Q2ce(Model, criterion, train_X, train_Y, labels, attributes, columns, numerical_attributes)
        return
    elif args.q == '2d':
        Q2d(criterion, train_X, train_Y, labels, attributes, columns, numerical_attributes, test_X, test_Y)
        return 

    # Normal or Q-2a,2b
    N=args_num(args.num)
    for t in range(*N):
        # AdaBoost
        if args.model.startswith('A'):
            Model = AdaBoost(num_estimators = t, criterion = criterion)
        # BaggedTrees
        elif args.model.startswith('B'):
            Model = BaggedTrees(num_trees=t, criterion=criterion, sample_ratio=args.ratio)
        # RandomForest
        elif args.model.startswith('R'):
            Model = RandomForest(num_trees=t, criterion=criterion, sample_ratio=args.ratio, 
            num_attributes=args.nattr)
    
        print(f"\t Model: {Model.__class__.__name__} \t num_tree: {t}")

        # train model
        if dotest:
            Model.train(train_X,
                    train_Y,
                    labels,
                    attributes,
                    columns,
                    numerical_attributes,
                    test_X,test_Y)

        else:
            Model.train(train_X,
                    train_Y,
                    labels,
                    attributes,
                    columns,
                    numerical_attributes)
        
            Model.evaluate(test_X,test_Y,verbose=True)
        print("="*70)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='../Data', help='Directory of Data folder. (Default: ../Data, ex: /path/to/Data)')
    parser.add_argument('--data', type=str, default='bank', help='Choose Dataset: car, bank. (Default: bank)')
    parser.add_argument('--model', type=str, default='AdaBoost', help='Choose: Ensemble Model. (Default: AdaBoost, BaggedTrees')
    parser.add_argument('-N','--num', type=str, default="50", help='Set number or range of # trees . (Default: 50, ex: 50, 1-50)')
    parser.add_argument('--ratio', type=float, default=1, help='Set ratio of boostrapping. (Default: 1)')
    parser.add_argument('--seed', type=float, default=None, help='Set random seed for boostrapping. (Default: None)')
    parser.add_argument('--nattr', type=int, default=2, help='Set bootstraping number of attribute for Randomforest. (Default: 2)')
    parser.add_argument('-q', type=str, default='', help='Choose question number. (options: 2a, 2b, 2c, 2d, 2e)')
        
    args = parser.parse_args()
    if args.q=='2a':
        args.model ='AdaBoost'
        args.num='1-500'
    elif args.q=='2b':
        args.model ='BaggedTrees'
        args.num='1-500'
    elif args.q=='2c':
        args.model ='BaggedTrees'
    elif args.q=='2d':
        args.model = 'RandomForest'
    elif args.q=='2e':
        args.model = 'RandomForest'

    main(args)