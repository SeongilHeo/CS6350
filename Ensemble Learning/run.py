from utils import *
from ID3 import ID3
from Ensemble import AdaBoost
import argparse

def main(args):
    DIR = f'./{args.dir}/{args.data}/'
    file_train = "train.csv"
    file_test = "test.csv"
    file_desc = "data-desc.txt"

    # load train csv
    csv_train=DIR+file_train
    train_data, train_labels = load_data(csv_train)

    # load test csv
    csv_test=DIR+file_test
    test_data, test_labels = load_data(csv_test)

    # load data desc
    labels, attributes, categorical_attributes, numerical_attributes, missing_attributes, columns = load_data_desc(DIR+file_desc)

    if numerical_attributes:
        train_data = preprosess_numeric(train_data, numerical_attributes, columns)
        test_data = preprosess_numeric(test_data, numerical_attributes, columns)

    criterion = 'information_gain'
    depth = 1

    for t in range(1,501):
        ada=AdaBoost(num_estimators = t, criterion = 'information_gain')

        ada.train(train_data,
                    train_labels,
                    labels,
                    attributes,
                    columns,
                    numerical_attributes)
        

        # train data
        train_acc = ada.evaluate(train_data, train_labels)
        test_acc = ada.evaluate(train_data, test_labels)
        print(t,train_acc,test_acc)
        print("-"*70)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='car', help='Choose Dataset: car, bank. (Default: car)')
    parser.add_argument('--dir', type=str, default='Data', help='Directory of Data folder. (Default: ./Data, ex: /path/to/Data)')
    parser.add_argument('-T','--tree', action='store_true', help='Enable this flag to visualize the tree')

    args = parser.parse_args()
    main(args)