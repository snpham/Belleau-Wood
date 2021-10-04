import numpy as np
from numpy.core.fromnumeric import swapaxes
import pandas as pd


def ch8_example():
    text = 'Janet will back the bill'
    text = text.split(' ')
    A_table = pd.read_csv('datasets/markov_A.txt', header=0, index_col=0, 
                          encoding='utf-8', sep='\s+', engine='python', skipinitialspace=True)

    B_table = pd.read_csv('datasets/markov_B.txt', header=0, index_col=0, 
                          encoding='utf-8', sep='\s+', engine='python', skipinitialspace=True)
    # states = tags

    tags = A_table.columns
    pi = A_table.iloc[0]
    A_table = A_table.to_numpy()
    B_table = B_table.to_numpy()

    # numpy method
    TEXS = len(text)
    TAGS = len(tags)
    viterbi = np.zeros(shape=(TAGS, TEXS))
    backpointer = np.zeros(shape=(TAGS, TEXS))

    for ii, tag in enumerate(tags):
        viterbi[ii, 0] = pi[ii]*B_table[ii][0]
        backpointer[ii, 0] = 0
    # print(viterbi)

    v_path = []
    prev_idx = 0
    for tex in range(1, TEXS):
        values = []
        for tag in range(0,TAGS): #TAGS
            # print(viterbi[prev_idx][tex-1], A_table[prev_idx+1][tag], B_table[tag][tex], '=',
                # viterbi[prev_idx][tex-1]*A_table[prev_idx+1][tag]*B_table[tag][tex])
            
    #         vt = viterbi[tag][tex-1] * A_table[tag+1][tag] * B_table[tag][tex]
            values.append(viterbi[prev_idx][tex-1]*A_table[prev_idx+1][tag]*B_table[tag][tex])
    #         viterbi[tag,tex] = vt
    #     v_j = max(values)
    #     v_path.append(v_j)
        # print(values)
        result = np.where(values == max(values))
        # print(result)
        prev_idx = result[0][0]
        viterbi[prev_idx][tex] = max(values)
        # print(prev_idx)
    # print(viterbi)
    viterbi = pd.DataFrame(viterbi)
    print(viterbi)


if __name__ == '__main__':
    pass


    # practice quiz 2
    A_table = pd.read_csv('datasets/amat_abc.txt', header=0, index_col=0, 
                          encoding='utf-8', sep='\s+', engine='python', skipinitialspace=True)

    B_table = pd.read_csv('datasets/bmat_abc.txt', header=0, index_col=0, 
                          encoding='utf-8', sep='\s+', engine='python', skipinitialspace=True)
    # states = tags
    tags = A_table.columns
    pi = A_table.iloc[0]
    A_table = A_table.to_numpy()
    B_seq = B_table.columns
    B_table = B_table.to_numpy()
    print(B_seq)
    print(B_table)

    # numpy method
    text = 'c a'
    text = text.split(' ')
    TEXS = len(text)
    TAGS = len(tags)
    viterbi = np.zeros(shape=(TAGS, TEXS))
    viterbi_all = np.zeros(shape=(TAGS, TEXS))
    backpointer = np.zeros(shape=(TAGS, TEXS))

    # here is the initiation of the viterbi matrix
    start_idx = np.where(B_seq == text[0])[0][0]
    for ii, tag in enumerate(tags):
        viterbi[ii, 0] = pi[ii]*B_table[ii][start_idx]
        viterbi_all[ii, 0] = pi[ii]*B_table[ii][start_idx]
        backpointer[ii, 0] = 0

    # we start at the second 'word'
    print(viterbi)
    v_path = []
    prev_idx = 0
    for tex, seq in zip(range(1, TEXS), text[1:]):
        print(tex, seq)
        seq_idx = np.where(B_seq == seq)[0][0]
        print(seq_idx)
        values = []
        for tag in range(0,TAGS): #TAGS
            print(viterbi[prev_idx][tex-1], A_table[prev_idx+1][tag], B_table[tag][seq_idx], '=',
                viterbi[prev_idx][tex-1]*A_table[prev_idx+1][tag]*B_table[tag][seq_idx])
            
    #         vt = viterbi[tag][tex-1] * A_table[tag+1][tag] * B_table[tag][tex]
            val = viterbi[prev_idx][tex-1]*A_table[prev_idx+1][tag]*B_table[tag][seq_idx]
            values.append(val)
            viterbi_all[tag][tex] = val
    #         viterbi[tag,tex] = vt
    #     v_j = max(values)
    #     v_path.append(v_j)
        print(values)
        result = np.where(values == max(values))
        print(result)
        prev_idx = result[0][0]
        viterbi[prev_idx][tex] = max(values)
        print(prev_idx)
    # print(viterbi)
    viterbi = pd.DataFrame(viterbi)
    print(viterbi)
    viterbi_all = pd.DataFrame(viterbi_all)
    print(viterbi_all)
