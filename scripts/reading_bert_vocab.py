import os
import urllib.request
from nltk.stem import PorterStemmer
stemmerporter = PorterStemmer()


# must run in linux terminal

if __name__ == '__main__':
    
    pass

    os.system('wc datasets/bert/BERT-vocab.txt')

    os.system("grep -v '^\[' < datasets/bert/BERT-vocab.txt |sort > datasets/bert/bertv1_nobrackets.txt")
    os.system('wc datasets/bert/bertv1_nobrackets.txt')

    os.system("grep -v '^.$' < datasets/bert/bertv1_nobrackets.txt > datasets/bert/bertv2_nosinglechar.txt")
    os.system('wc datasets/bert/bertv2_nosinglechar.txt')

    os.system("grep -v '^##' < datasets/bert/bertv2_nosinglechar.txt > datasets/bert/bertv3_nohash.txt")
    os.system('wc datasets/bert/bertv3_nohash.txt')

    os.system("egrep -v '[[:digit:]]+' < datasets/bert/bertv3_nohash.txt > datasets/bert/bertv4_nonumbers.txt")
    os.system('wc datasets/bert/bertv4_nonumbers.txt')

    url = 'https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt'
    dictionary = urllib.request.urlopen(url).read().decode('utf-8')
    dictionary = [f'{b}_' for b in dictionary.splitlines()]

    bertv5_spellcheck = []
    with open('datasets/bert/bertv4_nonumbers.txt', 'r') as readfile, \
        open('datasets/bert/bertv5_spellcheck.txt', 'w') as writefile:

        a = readfile.read().splitlines()
        for vocab in a:
            if f'{vocab}_' in dictionary:
                writefile.write(f'{vocab}\n')
    os.system('wc datasets/bert/bertv5_spellcheck.txt')

    with open('datasets/bert/bertv5_spellcheck.txt', 'r') as readfile, \
        open('datasets/bert/bertv6_morph.txt', 'w') as writefile:

        a = readfile.read().splitlines()
        for vocab in a:
            stem = stemmerporter.stem(vocab)
            writefile.write(f'{stem}\n')
    os.system('wc datasets/bert/bertv6_morph.txt')

    os.system("sort datasets/bert/bertv6_morph.txt | uniq > datasets/bert/bertv7_final.txt")
    os.system('wc datasets/bert/bertv7_final.txt')
