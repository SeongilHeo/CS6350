from utils import *
from ID3 import ID3
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
    try:
        labels, attributes, numerical_attributes, columns = load_data_desc(DIR+file_desc)
    except:
        file_desc="./data-desc.txt"
        labels, attributes, numerical_attributes, columns = load_data_desc(file_desc)



    if numerical_attributes:
        train_data = preprosess_numeric(train_data, numerical_attributes,columns)
        test_data = preprosess_numeric(test_data, numerical_attributes, columns)

    if args.miss:
        keys=[] # "job", "education", "contact","poutcome"]
        for name, values in attributes.items():
            if values and "unknown" in values:
                keys.append(name)
        for key in keys:
            preprosess_miss(train_data, columns[key])    
            preprosess_miss(test_data, columns[key])    

    # ID3tree.train(train_data,
    #             train_labels,
    #             attributes,
    #             columns,
    #             numerical_attributes)
    
    # print(f"[INFO] Success train ID3 with {args.data} dataset.")

    # ID3tree.predict(test_data)
    # ID3tree.evaluate(test_data, test_labels)

    train_avg=[]
    test_avg=[]
    total_avg=[]

    if args.criterion:
        criterions = [args.criterion]
    else:
        criterions = ['information_gain','majority_error','gini_index']

    if args.depth:
        max_depth_range = [args.depth]
    else:
        max_depth_range = range(1,len(attributes)+1)


    for depth in max_depth_range:
        # print(f"{depth}", end="")
        for criterion in criterions:
            ID3tree=ID3(max_depth = depth, criterion = criterion)

            ID3tree.train(train_data,
                        train_labels,
                        attributes,
                        columns,
                        numerical_attributes)
            
            if args.tree:
                ID3tree.visualization()

            # train data
            train_hit = ID3tree.evaluate(train_data, train_labels, verbose=False)
            train_acc = train_hit / len(train_labels)
            train_err = 1 - train_acc

            # train test
            test_hit = ID3tree.evaluate(test_data, test_labels, verbose=False)
            test_acc = test_hit / len(test_labels)
            test_err = 1 - test_acc

            # train+test data
            total_hit = ID3tree.evaluate(train_data+test_data, train_labels+test_labels, verbose=False)
            total_acc = total_hit / (len(train_labels) + len(test_labels))
            total_err = 1 - total_acc

            train_avg.append(train_err)
            test_avg.append(test_err)
            total_avg.append(total_err)
            
            # train_avg.append(train_acc)
            # test_avg.append(test_acc)
            # avg.append(acc)

            print(f"Criterion: {criterion}, Depth: {depth}")
            print(f"\t train error: {train_err:.3f}, test error: {test_err:.3f}, total_error: {total_err:.3f}")
            
            # for latex
            # print(f" & {train_err:.3f} & {test_err:.3f} & {total_err:.3f}", end=" ")
            # print(f"& {train_acc:.3f} & {test_acc:.3f} & {acc:.3f}", end=" ")
        # print("\\ \hline")
        print("-"*70)

    # print("avg.",end=" ")

    for i in range(len(criterions)):
        f_train_avg = sum(train_avg[i::len(criterions)]) / len(max_depth_range)
        f_test_avg = sum(test_avg[i::len(criterions)]) / len(max_depth_range)
        f_total_avg = sum(total_avg[i::len(criterions)]) / len(max_depth_range)
        print(criterions[i], end="\t")
        print(f"Avg. train: {f_train_avg:.3f} & test: {f_test_avg:.3f} & total: {f_total_avg:.3f}")
        # for latex
        # print(f"& {f_train_avg:.3f} & {f_test_avg:.3f} & {f_total_avg:.3f}", end=" ")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='car', help='Choose Dataset: car, bank. (Default: car)')
    parser.add_argument('--dir', type=str, default='Data', help='Directory of Data folder. (Default: ./Data, ex: /path/to/Data)')
    parser.add_argument('-M','--miss', action='store_true', help='Enable this flag to handle "unknown" values.')
    parser.add_argument('-T','--tree', action='store_true', help='Enable this flag to visualize the tree')
    parser.add_argument('-D','--depth', type=str, help='Set ID3 max depth. (Default: 1 to Max)')
    parser.add_argument('-C','--criterion', type=str, help='Set criterion for inpurity: information_gain, majority_error, gini_index. (Default: All)')

    args = parser.parse_args()
    main(args)