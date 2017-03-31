import numpy as np
import cPickle as pickle


def load_wordvec(filename, vecdim):
    print("load wordVec...")
    wordvec_dic = {}
    wordvec_lst = []
    wordvec_dic['UNK']=0
    wordvec_lst.append(np.random.randn(vecdim) * 0.5)
    with open(filename, 'r') as wordvec_file:
        for i, line in enumerate(wordvec_file):
            content = line.split()
            wordvec_dic[content[0]] = i + 1
            wordvec_lst.append(np.array(map(float, content[1:]), dtype='float32'))
            if 0 == i % 100000:
                print i
    #i=i+1
    #wordvec_dic['UNK']=i
    print i+1
    #wordvec_lst.append(np.random.randn(vecdim) * 0.5)
    wordvec_lst = np.array(wordvec_lst)
    # wordvec_lst=wordvec_lst.T
    return wordvec_lst,wordvec_dic

