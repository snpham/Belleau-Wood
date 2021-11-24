from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

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
from pprint import pprint as pp


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

    # first sentence
    strng = df.iloc[0:new_sentences_idx[0], 1].to_list()
    s = ' '.join(strng)
    print(s)


    from transformers import AutoModelForTokenClassification, AutoTokenizer
    import torch

    model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    label_list = [
        "O",       # Outside of a named entity
        "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
        "I-MISC",  # Miscellaneous entity
        "B-PER",   # Beginning of a person's name right after another person's name
        "I-PER",   # Person's name
        "B-ORG",   # Beginning of an organisation right after another organisation
        "I-ORG",   # Organisation
        "B-LOC",   # Beginning of a location right after another location
        "I-LOC"    # Location
    ]

    sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very" \
            "close to the Manhattan Bridge."

    # Bit of a hack to get the tokens with the special tokens
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(s)))
    inputs = tokenizer.encode(s, return_tensors="pt")

    outputs = model(inputs)[0]
    predictions = torch.argmax(outputs, dim=2)

    pp([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())])