from utils import *
from ID3 import ID3
import argparse

parser = argparse.ArgumentParser()


parser.add_argument('arg1', type=str, help='"car" / "bank"')
parser.add_argument('arg2', type=str, nargs='?', default="N", help='hanlde unknown value T/F (optional)')

parser.add_argument('arg3', type=str, help='"data file dir"')

args = parser.parse_args()

dataset =  args.arg1
miss =  args.arg2
dir =  args.arg3


DIR = f'./{dir}/{dataset}/'
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



ID3tree=ID3()

if numerical_attributes:
    train_data = preprosseing(train_data,numerical_attributes, columns)
    test_data = preprosseing(test_data,numerical_attributes, columns)

if miss == "T":
    keys=["job", "education", "contact","poutcome"]
    for key in keys:
        missing(train_data,columns[key])    
        missing(test_data,columns[key])    

ID3tree.train(train_data,
              train_labels,
              attributes,
              columns,
              numerical_attributes)

# ID3tree.predict(test_data)
ID3tree.evaluate(test_data, test_labels)

train_avg=[]
test_avg=[]
avg=[]

criterion = ['information_gain','majority_error','gini_index']

for d in range(1,17):
    print(f"{d}", end="")
    for c in criterion:
        ID3tree=ID3(max_depth = d, criterion = c)

        ID3tree.train(train_data,
                    train_labels,
                    attributes,
                    columns,
                    numerical_attributes)
        
        # train data
        train_hit = ID3tree.evaluate(train_data, train_labels, verbose=False)
        train_acc = train_hit/len(train_labels)
        train_err = 1 - train_acc

        # train test
        test_hit = ID3tree.evaluate(test_data, test_labels, verbose=False)
        test_acc = test_hit/len(test_labels)
        test_err = 1 - test_acc

        # train+test data
        hit = ID3tree.evaluate(train_data+test_data, train_labels+test_labels, verbose=False)
        acc = hit / (len(train_labels)+len(test_labels))
        err = 1 - acc

        train_avg.append(train_err)
        test_avg.append(test_err)
        avg.append(err)
        
        # train_avg.append(train_acc)
        # test_avg.append(test_acc)
        # avg.append(acc)

        print(f"Criterion: {c}, Depth: {d}")
        print(f"\t train error: {train_err:.3f}, test error: {test_err:.3f}, total_error: {err:.3f}")
        
        # for latex
        # print(f" & {train_err:.3f} & {test_err:.3f} & {err:.3f}", end=" ")
        # print(f"& {train_acc:.3f} & {test_acc:.3f} & {acc:.3f}", end=" ")
    # print("\\ \hline")

print("avg.",end=" ")

for i in range(3):
    f_train_avg=sum(train_avg[i::3])/6
    f_test_avg=sum(test_avg[i::3])/6
    f_avg=sum(avg[i::3])/6
    print(criterion[i])
    print(f"Avg. train: {f_train_avg:.3f} & test: {f_test_avg:.3f} & total: {f_avg:.3f}", end=" ")
    # for latex
    # print(f"& {f_train_avg:.3f} & {f_test_avg:.3f} & {f_avg:.3f}", end=" ")