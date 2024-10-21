import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def draw_bias_var(Bias,full_Bias, Var, full_Var):
    # Bias
    x = np.arange(len(Bias))
    b=np.array(Bias)
    b1=np.array(full_Bias) * (-1)

    plt.figure(figsize=(10, 6))
    plt.bar(x, b, color='blue', alpha=0.7, label='Bias (Single)')
    plt.bar(x, b1, color='red', alpha=0.7, label='Bias (Bagged)')  
    plt.axhline(0, color='black', linewidth=1) 

    plt.ylim(-3, 3)

    yticks = plt.gca().get_yticks()  
    plt.yticks(yticks, np.abs(yticks)) 

    plt.xlabel('Data Points')
    plt.ylabel('Bias')
    plt.title('Bias Visualization')
    plt.legend()
    plt.grid(True)
    current_time = datetime.now()
    plt.savefig(f"./Img/Bias_{current_time.strftime('%Y%m%d_%H%M%S')}.png")

    # Variance 
    x = np.arange(len(Var))
    v = np.array(Var)
    v1 = np.array(full_Var) * (-1)

    plt.figure(figsize=(10, 6))
    plt.bar(x, v, color='blue', alpha=0.7, label='Variance (Single)')
    plt.bar(x, v1, color='red', alpha=0.7, label='Variance (Bagged)')
    plt.axhline(0, color='black', linewidth=1)

    plt.ylim(-1, 1)

    yticks = plt.gca().get_yticks()  
    plt.yticks(yticks, np.abs(yticks)) 

    plt.xlabel('Data Points')
    plt.ylabel('Variance')
    plt.title('Variance Visualization')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./Img/Var_{current_time.strftime('%Y%m%d_%H%M%S')}.png")

    print(f"[Info] The graphs are saved at ./Img/Var_{current_time.strftime('%Y%m%d_%H%M%S')}.png")


def draw_randomforest(result):
    err_2 = [r[0] for r in result]
    err_4 = [r[1] for r in result]
    err_6 = [r[2] for r in result]

    train_2 = np.array([err[0] for err in err_2])
    test_2 =  np.array([err[1] for err in err_2])
    train_4 = np.array([err[0] for err in err_4])
    test_4 =  np.array([err[1] for err in err_4])
    train_6 = np.array([err[0] for err in err_6])
    test_6 =  np.array([err[1] for err in err_6])

    x = np.arange(len(result))

    plt.figure(figsize=(10, 6))
    plt.plot(x, train_2, alpha=0.7, label='#attr:2 (train)')
    plt.plot(x,  test_2, alpha=0.7, label='#attr:2 (test)')  
    plt.plot(x, train_4, alpha=0.7, label='#attr:4 (train)')
    plt.plot(x,  test_4, alpha=0.7, label='#attr:4 (test)')  
    plt.plot(x, train_6, alpha=0.7, label='#attr:6 (train)')
    plt.plot(x,  test_6, alpha=0.7, label='#attr:6 (test)')  
    # plt.axhline(0, color='black', linewidth=1) 

    # plt.ylim(-3, 3)

    # yticks = plt.gca().get_yticks()  
    # plt.yticks(yticks, np.abs(yticks)) 

    plt.xlabel('Data Points')
    plt.ylabel('error')
    plt.title('Randomforest Visualization')
    plt.legend()
    plt.grid(True)
    current_time = datetime.now()
    plt.savefig(f"./Img/RandomForest_Q2d_{current_time.strftime('%Y%m%d_%H%M%S')}.png")

    print(f"[Info] The graphs are saved at ./Img/RandomForest_Q2d_{current_time.strftime('%Y%m%d_%H%M%S')}.png")
