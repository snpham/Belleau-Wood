import numpy as np
import pandas as pd


def laplace_smoothing(count_wi, count_w, V):
    return (count_wi + 1) / (count_w + V)

if __name__ == '__main__':
    pass

    S_table = pd.read_csv('datasets/smoothing.txt', header=0, index_col=0, 
                          encoding='utf-8', sep='\s+', engine='python', 
                          skipinitialspace=True)
    print(S_table)
    S_table = S_table.to_numpy()

    V = 1446
    k = 1
    Snew_table = np.zeros(shape=(len(S_table), len(S_table[0])))
    # print(np.shape(Snew_table))
    for ii in range(0, len(S_table[0])):
        for jj in range(0, len(S_table)):
            # print(ii, jj)

            count_wi = S_table[ii][jj]
            count_w = np.sum(S_table[:][ii])
            # print(Snew_table[jj][ii])
            Snew_table[jj][ii] = laplace_smoothing(count_wi, count_w, V)
    print(Snew_table)
    

    