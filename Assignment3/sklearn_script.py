import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


if __name__ == '__main__':
    pass

    data = []
    # load data
    with open('S21-gene-train.txt', 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            line = line.split('\t')
            # if line == ['']:
            #     line = '__'
            data.append(line)

    # make a dataframe
    df = pd.DataFrame(data=data, columns=['Sequence', 'Word', 'Tag'])
    
    # debug
    print(df.head(20))
    
    # indices of new sentence rows
    new_sentences_idx = np.array(df[df['Sequence'] == ''].index)
    # print(df[df['Sequence'] == ''])
    print(new_sentences_idx)

    # get B, I, O tag counts
    df_counts = df.groupby('Tag').size().reset_index(name='counts')
    print(df_counts)

    # turn 
    X = df.drop('Tag', axis=1)
    v = DictVectorizer(sparse=False)
    X = v.fit_transform(X.to_dict('records'))
    print(X, X.shape)
    y = df.Tag.values
    print(y)

    classes = np.unique(y[y != None])
    print(classes)

