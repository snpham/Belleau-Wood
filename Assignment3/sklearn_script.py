import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer


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
            #     line = '__'
            data.append(line)

    df = pd.DataFrame(data=data, columns=['Sequence', 'Word', 'Tag'])
    
    print(df.head(20))
    # print(data)
    
    new_sentences_idx = np.array(df[df['Sequence'] == ''].index)
    print(df[df['Sequence'] == ''])
    print(new_sentences_idx)

    df_counts = df.groupby('Tag').size().reset_index(name='counts')
    print(df_counts)

    X = df.drop('Tag', axis=1)
    v = DictVectorizer(sparse=False)
    X = v.fit_transform(X.to_dict('records'))
    print(X)
    y = df.Tag.values
    print(y)

    classes = np.unique(y[y != None])
    print(classes)
