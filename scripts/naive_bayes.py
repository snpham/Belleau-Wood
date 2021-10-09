import numpy as np
import pandas as pd
from collections import Counter


def get_word_probability(word, counts, counts_vocab):
    """
    :param word: the word to count
    :param counts: total count of words in the class
    :param counts_vocab: total unique words in data set
    """
    return (counts[word]+1) / (sum(counts.values())+counts_vocab)


def get_counts(doc):
    counts_pos = Counter()
    counts_neg = Counter()
    for row in doc.itertuples():
        for item in row.document:
            if row.cat == '-':
                counts_neg[item] += 1
            if row.cat == '+':
                counts_pos[item] += 1
    # print(counts_pos)
    # print(counts_neg)
    counts_total = counts_pos + counts_neg
    counts_vocab =  len(counts_total.keys())

    return counts_pos, counts_neg, counts_total, counts_vocab


def rm_unkwn_words(test_line, counts_total):
    words = []
    for word in test_line.split(' '):
        if word in counts_total.keys():
            words.append(word)
    return np.array(words)


def accuracy(df):
    return (df.iloc[0,0]+df.iloc[1,1]) / (df.iloc[2,2])


def precision(df):
    return df.iloc[0,0] / (df.iloc[0,0] + df.iloc[1,0])


def recall(df):
    return df.iloc[0,0] / (df.iloc[0,0] + df.iloc[0,1])



if __name__ == '__main__':
    pass

    doc = pd.read_csv('datasets/naive_bayes.csv', skipinitialspace=True)
    doc['document'] = doc['document'].str.split()

    counts_pos, counts_neg, counts_total, counts_vocab = get_counts(doc)

    test_line = 'predictable with no fun'
    words = rm_unkwn_words(test_line, counts_total=counts_total)

    p_prior_pos = len(doc[doc.cat == '+'])/len(doc.cat)
    p_prior_neg = len(doc[doc.cat == '-'])/len(doc.cat)
    p_pos = []
    p_neg = []
    for word in test_line.split(' '):
        if word in counts_total.keys():
            p_pos.append(get_word_probability(word=word, counts=counts_pos, 
                                            counts_vocab=counts_vocab))
            p_neg.append(get_word_probability(word=word, counts=counts_neg, 
                                            counts_vocab=counts_vocab))

    p_pos = np.prod(np.array(p_pos))*p_prior_pos
    p_neg = np.prod(np.array(p_neg))*p_prior_neg
    assert np.allclose(p_pos, 3.280167288531715e-05)
    assert np.allclose(p_neg, 6.106248727864848e-05)
    

    data = pd.read_csv('datasets/classifier_ex.csv', skipinitialspace=True, index_col=0)
    acc = accuracy(data)
    prec = precision(data)
    rec = recall(data)
    assert np.allclose([acc, prec, rec], [0.965, 0.3913, 0.3], rtol=1e-2)
    