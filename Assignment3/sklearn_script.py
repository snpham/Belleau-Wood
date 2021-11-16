import pandas as pd
import numpy as np


if __name__ == '__main__':
    pass

    # df = pd.read_csv('S21-gene-train.txt')

    data = []
    # load data
    with open('S21-gene-train.txt', 'r') as f:
        for line in f.readlines():
            # print(line)
            line = line.replace('\n', '')
            line = line.split('\t')
            # if line == ['']:
            #     print(line)
            data.append(line)

    df = pd.DataFrame(data=data)

    print(df.head(20))
    # print(data)
    
    new_sentences_idx = np.array(df[df[0] == ''].index)
    print(df[df[0] == ''])
    print(new_sentences_idx)

    