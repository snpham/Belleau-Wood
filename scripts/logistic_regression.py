import numpy as np
import pandas as pd
from collections import Counter
import re


def sigmoid(z):
    # print('z value:', z)
    return 1 / (1+ np.exp(-z))


def z_value(w, x, b):
    return np.dot(w, x) + b


def cross_entropy(w, x, b, y):
    sigma = sigmoid(z_value(w, x, b))
    L_CE = -(y * np.log(sigma) + (1-y)*np.log(1-sigma))
    return L_CE


def cross_entropy_loss(y_est, y):
    return -(y * np.log(y_est) + (1-y)*np.log(1-y_est))


def gradient(x, y, w, b):
    z =  z_value(w, x, b)
    del_L = []

    del_L = [(sigmoid(z)-y)*xi for xi in x]

    del_b = sigmoid(z)-y
    del_L.append(del_b)

    return np.array(del_L)


def update_weight(w, b, nu, grad):
     return np.subtract(np.hstack((w, b)), nu*grad)


def stochastic_grad(x, y, w, b, nu):
    grad = gradient(x, y, w, b)
    return np.subtract(np.hstack((w, b)), nu*grad)
    

def softmax(z):
    """requires z to be a vector with a value for each in the 
    K classes.
    """
    sum_z = sum(np.exp(zi) for zi in z)
    return [np.exp(zi)/sum_z for zi in z]


def run_tests():
    # testing sigmoid
    w = np.array([2.5, -5.0, -1.2, 0.5, 2.0, 0.7])
    x = np.array([3, 2, 1, 3, 0, 4.15])
    b = 0.1
    P_pos = sigmoid(z=z_value(w, x, b))
    print(z_value(w, x, b))
    P_neg = 1 - P_pos
    assert np.allclose([P_pos, P_neg], [0.69104, 0.30895], rtol=1e-2)


    y = 1
    c_ent = cross_entropy(w, x, b, y)
    print(c_ent)
    assert np.allclose(c_ent, 0.369, rtol=1e-2)

    
    # testing gradient and stochastic gradient
    x = [3,2]
    y = 1
    w = [0, 0]
    b = 0
    nu = 0.1
    grad = gradient(x, y, w, b)
    assert np.allclose(grad, [-1.5, -1.0, -0.5], rtol=1e-2 )

    stoch = stochastic_grad(x, y, w, b, nu)
    assert np.allclose(stoch, [0.15, 0.1, 0.05], rtol=1e-2 )


    # testing softmax
    z = [0.6, 1.1, -1.5, 1.2, 3.2, -1.1]
    sm = softmax(z)
    assert np.allclose(sm, 
        [0.05482, 0.09039, 0.006713, 0.0998, 0.7381, 0.01001], rtol=1e-2)


    # test from lecture 9/21 - not sure how he got updated weights
    w = np.array([2.5, -5.0, -1.2, 0.5, 2.0, 0.7])
    x = np.array([3, 2, 1, 3, 0, 4.15])
    b = 0.1
    y = 1
    P_pos = sigmoid(z=z_value(w, x, b))
    assert np.allclose(P_pos, 0.69104, rtol=1e-2)

    grad = gradient(x, y, w, b)
    assert np.allclose(grad, [-0.9268, -0.6179, -0.3089, -0.9268, -0., -1.282, -0.3089], rtol=1e-2)

    nu = 1
    w_new = update_weight(w, b, nu, grad)
    assert np.allclose(w_new, [3.426, -4.382, -0.891, 1.426, 2., 1.982, 0.408], rtol=1e-2)
    print(w_new[:-1])
    z = z_value(w=w_new[:-1], x=x, b=b)
    sig = sigmoid(z=z)
    print(sig)

    # test from lecture 9/21
    w = np.array([0., 0., 0., 0., 0., 0.])
    x = np.array([3, 2, 1, 3, 0, 4.15])
    b = 0.1
    y = 1
    P_pos = sigmoid(z=z_value(w, x, b))
    assert np.allclose(P_pos, 0.5249, rtol=1e-2)

    # grad = gradient(x, y, w, b)
    # # assert np.allclose(grad, [-0.9268, -0.6179, -0.3089, -0.9268, -0., -1.282, -0.3089], rtol=1e-2)
    # print(grad)
    # nu = 1
    # w_new = update_weight(w, b, nu, grad)
    # # assert np.allclose(w_new, [3.426, -4.382, -0.891, 1.426, 2., 1.982, 0.408], rtol=1e-2)
    # print(w_new[:-1])
    # z = z_value(w=w_new[:-1], x=x, b=b)
    # sig = sigmoid(z=z)
    # print(sig)


    # x = [-3, 1, 4, 1]
    # w = [1, 1, 1, 1]
    # y = 1
    # b = 0
    # grad = gradient(x, y, w, b)
    # print(grad)

    l_ce = cross_entropy_loss(y_est=0.4, y=1)
    print(l_ce)


def extraction(review, pos_words, neg_words, pronouns):
    counts = Counter()

    review = review.lower()

    if '!' in review:
        counts['x5'] += 1

    review = re.sub(r'[^\w\-\s]','', review)
    # print(review)
    words = review.split()
    words = [f'_{w}_' for w in words]

    ct = 0
    for w in words:
        if w in pos_words:
            counts['x1'] += 1
        if w in neg_words:
            counts['x2'] += 1
        if w in pronouns:
            counts['x4'] += 1
        if w == '_no_' and ct == 0:
            counts['x3'] += 1
            ct = 1

    return [counts, np.log(len(words))]


if __name__ == '__main__':
    pass
    # run_tests()

    # get reviews
    review_file_pos = np.loadtxt('datasets/assignment2/hotelPosT-train.txt', 
                                 delimiter='\t', dtype='str', encoding="utf8")
    review_file_neg = np.loadtxt('datasets/assignment2/hotelNegT-train.txt', 
                                 delimiter='\t', dtype='str', encoding="utf8")

    # get word semantics
    pos_words = np.loadtxt('datasets/assignment2/positive-words.txt', 
                           delimiter='\n', dtype='str')
    neg_words = np.loadtxt('datasets/assignment2/negative-words.txt', 
                           delimiter='\n', dtype='str')
    pronouns = np.loadtxt('datasets/assignment2/pronouns.txt', 
                           delimiter='\n', dtype='str')
    pos_words = [f'_{w}_' for w in pos_words]
    neg_words = [f'_{w}_' for w in neg_words]    
    pronouns = [f'_{w}_' for w in pronouns]

    # extract positive and negative words
    reviews_pos = review_file_pos[:, 1]
    extracts_pos = [extraction(rev, pos_words, neg_words, pronouns) for rev in reviews_pos]
    vectors_pos = [[ex['x1'], ex['x2'], ex['x3'], ex['x4'], ex['x5'], lnw] for ex, lnw in extracts_pos]
    # print(vectors_pos)
    reviews_neg = review_file_neg[:, 1]
    extracts_neg = [extraction(rev, pos_words, neg_words, pronouns) for rev in reviews_neg]
    vectors_neg = [[ex['x1'], ex['x2'], ex['x3'], ex['x4'], ex['x5'], lnw] for ex, lnw in extracts_neg]
    # print(vectors_neg)

    # concatenate positive and negative vectors
    vectors = np.vstack([vectors_pos, vectors_neg]).tolist()
    review_ids = np.hstack([list(review_file_pos[:,0]), list(review_file_neg[:,0])])
    # print(vectors)
    # print(review_ids)

    # assuming bias=1, generate csv file of results
    b = 1
    with open('datasets/assignment2/pham-son-assgn2-part1.csv', 'w')as f:
        for id, vec in zip(review_ids, vectors):
            f.write(f'{id},{int(vec[0])},{int(vec[1])},{int(vec[2])},'
                    f'{int(vec[3])},{int(vec[4])},{round(vec[5],2)},{b}\n')

    w = [0,0,0,0,0,0]
    reviews_file = np.loadtxt('datasets/assignment2/pham-son-assgn2-part1.csv', 
                                 delimiter=',', dtype='str', encoding="utf8")
    # print(np.shape(reviews_file))

    ids = reviews_file[:,0]
    # print(ids)
    vectors = reviews_file[:,1:-1]
    bias = reviews_file[:,-1]
    # print(vectors)
    # print(bias)
    
    # nu = 1
    # gold_label = 1
    # stoch = stochastic_grad(vectors[0], gold_label, w, bias[0], nu)
    # print(stoch)
    # stoch = stochastic_grad(vectors[0], gold_label, stoch[:-1], stoch[-1], nu)
    # print(stoch)
    