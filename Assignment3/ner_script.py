import numpy as np
import pandas as pd


if __name__ == '__main__':
    
    pass


    # load data
    # data = np.loadtxt('S21-gene-train.txt', delimiter='\t', dtype='str', encoding="utf8")
    # data = pd.read_csv('S21-gene-train.txt', delimiter='\t', index_col=0)
    # print(data)

    # with open('S21-gene-train.txt', 'r') as f:
    #     for line in f.readlines():
    #         print(line)


    from transformers import pipeline

    # Initialize the NER pipeline
    ner = pipeline("ner")

    # Phrase
    phrase = "David helped Peter enter the building, where his house is located."

    # NER task
    ner_result = ner(phrase)

    # Print result
    print(ner_result)