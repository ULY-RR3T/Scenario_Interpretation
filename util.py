import numpy as np
import matplotlib.pyplot as plt

iteration_N = 10000

def argmax_true(list):
    for i in reversed(range(len(list))):
        if list[i] == True:
            return i
    return len(list) - 1

def argmin_true(list):
    for i in range(len(list)):
        if list[i] == True:
            return i
    return 0

def plot_result(result,q,save_fig=False):
    wks = list(range(1,result.shape[0] + 1))
    plt.figure()
    plt.plot(wks, result[:, 1], label='$Z^L median$', linewidth=2)
    plt.plot(wks, result[:, 2], '--',label='$Z^L mean$', linewidth=2)
    plt.plot(wks, result[:, 3], '--',label='$Z^U mean$', linewidth=2)
    plt.plot(wks, result[:, 4], label='$Z^U median$', linewidth=2)
    plt.fill_between(wks, result[:, 0], result[:, 5], color='b', alpha=0.15)
    plt.legend()
    plt.ylabel("DFRV")
    plt.xlabel("Weeks Since Prediction")
    plt.show()