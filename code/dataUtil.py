#!/usr/bin/env python
#coding=utf-8
import random
import sys
import math
import nltk
from loadWordVec import load_wordvec

class DataUtil:
    def __init__(self):
        pass
    
    def readFileLineByLine(self, path):
        result = []
        fd = open(path, 'r')
        for line in fd:
            line = line.replace('\r\n', '')
            result.append(line)
        
        return result
    
    def loadLabel(self, path, rel):
        label = self.readFileLineByLine(path)
        result = []
        for lab in label:
            lab = '' + lab.strip() + ''
            if rel == lab:
                result.append(1)
                continue
            result.append(0)
        return result

    def filter(self, arg1, arg2, label, minlength):
        result_arg1 = []
        result_arg2 = []
        result_label = []
        for i in xrange(len(arg1)):
            length = len(arg1[i]) * len(arg2[i])
            if length >= minlength:
                result_arg1.append(arg1[i])
                result_arg2.append(arg2[i])
                result_label.append(label[i])
        return result_arg1, result_arg2, result_label
      
    def loadData(self, rel, demension, wordVec):
        # 加载数据
        reload(sys)
        sys.setdefaultencoding('utf-8')
        if wordVec == 'glove':
           vector_filepath = '../vectors/all_1b_v'+ str(demension) +'_w2_ns5_100'
	else:
           vector_filepath = '../vectors/vectors.6B.'+str(demension)+'d.txt'  
        (wordVec, wordMap) = load_wordvec(vector_filepath, demension)
        (dev_arg1, dev_arg2, dev_label) = self.loadDataSet('dev', rel, wordMap)
        (test_arg1, test_arg2, test_label) = self.loadDataSet('test', rel, wordMap)
        (train_arg1, train_arg2, train_label) =self.load_traindata(rel, wordMap)
        posData_arg1 = []
        posData_arg2 = []
        negData_arg1 = []
        negData_arg2 = []
            
        for i in xrange(len(train_arg1)):
            if train_label[i] == 1:
                posData_arg1.append(train_arg1[i])
                posData_arg2.append(train_arg2[i])
            else:
                negData_arg1.append(train_arg1[i])
                negData_arg2.append(train_arg2[i])
        posLen = len(posData_arg1)
        negLen = len(negData_arg1)

        print 'after adjust: train all number:', len(train_arg1)
        print 'after adjust: pos all number:', posLen
        print 'after adjust: neg all number:', negLen
        return (wordMap, wordVec, train_arg1, train_arg2, train_label, dev_arg1, dev_arg2, dev_label, test_arg1, test_arg2, test_label)

    def load_traindata(self, rel, wordMap):
        (train_arg1, train_arg2, train_label) = self.loadDataSet('train', rel, wordMap)
        posData_arg1 = []
        posData_arg2 = []
        negData_arg1 = []
        negData_arg2 = []
            
        for i in xrange(len(train_arg1)):
            if train_label[i] == 1:
                posData_arg1.append(train_arg1[i])
                posData_arg2.append(train_arg2[i])
            else:
                negData_arg1.append(train_arg1[i])
                negData_arg2.append(train_arg2[i])
        posLen = len(posData_arg1)
        negLen = len(negData_arg1)
         
        # 正例比负例多  Exp + Ent时
        if posLen > negLen:
            data_idx = list(range(len(posData_arg1)))
            random.shuffle(data_idx)
            random.shuffle(data_idx)
            posData_arg1 = [posData_arg1[i] for i in data_idx]
            posData_arg2 = [posData_arg2[i] for i in data_idx]
            posData_arg1 = posData_arg1[:negLen]
            posData_arg2 = posData_arg2[:negLen]

            train_arg1 = []
            train_arg2 = []
            train_label = []
            for pos_arg1, pos_arg2, neg_arg1, neg_arg2 in zip(posData_arg1, posData_arg2, negData_arg1, negData_arg2):
                train_arg1.append(neg_arg1)
                train_arg2.append(neg_arg2)
                train_label.append(0)

                train_arg1.append(pos_arg1)
                train_arg2.append(pos_arg2)
                train_label.append(1)
            #pass

        else:
            # 反例比正例多 
            data_idx = list(range(len(negData_arg1)))
            random.shuffle(data_idx)
            random.shuffle(data_idx)
            negData_arg1 = [negData_arg1[i] for i in data_idx]
            negData_arg2 = [negData_arg2[i] for i in data_idx]
            negData_arg1 = negData_arg1[: posLen]
            negData_arg2 = negData_arg2[: posLen]

            train_arg1 = []
            train_arg2 = []
            train_label = [] 
            for pos_arg1, pos_arg2, neg_arg1, neg_arg2 in zip(posData_arg1, posData_arg2, negData_arg1, negData_arg2):
                train_arg1.append(neg_arg1)
                train_arg2.append(neg_arg2)
                train_label.append(0)

                train_arg1.append(pos_arg1)
                train_arg2.append(pos_arg2)
                train_label.append(1)


        data_idx = list(range(len(train_arg1)))
        random.shuffle(data_idx)
        random.shuffle(data_idx)
        train_arg1 = [train_arg1[i] for i in data_idx]
        train_arg2 = [train_arg2[i] for i in data_idx]
        train_label = [train_label[i] for i in data_idx]
        #print 'after adjust: train all number:', len(train_arg1)
        return train_arg1, train_arg2, train_label
    
    def adjust(self, loseLength, length, train_arg1, train_arg2, train_label, add_arg1, add_arg2, label):
        data_idx = list(range(len(add_arg1)))
        random.shuffle(data_idx)
        random.shuffle(data_idx)
        add_arg1 = [add_arg1[i] for i in data_idx]
        add_arg2 = [add_arg2[i] for i in data_idx]

        if loseLength > length:
            d = loseLength / length
            loseLength = loseLength - d * length
            for i in range(d):
                train_arg1.extend(add_arg1)
                train_arg2.extend(add_arg2)
                train_label.extend([label for x in range(length)])

        loseLength = loseLength % length
        train_arg1.extend(add_arg1[0: loseLength])
        train_arg2.extend(add_arg2[0: loseLength])
        train_label.extend([label for x in range(loseLength)])
        return train_arg1, train_arg2, train_label

    def loadDataSet(self, data_type, rel, wordMap):
        if rel == 'expansion':
            prefex = '../data/withEnt/'
        else: 
            prefex = '../data/withoutEnt/'

        arg1 = self.readFileLineByLine(prefex + data_type + '_arg1.txt')
        arg2 = self.readFileLineByLine(prefex + data_type + '_arg2.txt')
        label = self.loadLabel(prefex + data_type + '_label.txt', rel)
        arg1 = self.transform(arg1, wordMap)
        arg2 = self.transform(arg2, wordMap)
        arg1, arg2, label = self.filter(arg1, arg2, label, 10)
        return (arg1, arg2, label)
    
    def transform(self, data_set, wordMap):
        data_set = self.tokenize(data_set)
        #print data_set
        for i in xrange(len(data_set)):
            seq = []
            for word in data_set[i]:
                if word not in wordMap:
                    word = wordMap['UNK']
                else:
                    word = wordMap[word]
                seq.append(word)
            
            data_set[i] = seq
        return data_set

    
    def tokenize(self, data_set):
        for i in xrange(len(data_set)):
            data_set[i] = nltk.word_tokenize(data_set[i])
        return data_set

    
    def connectText(self, arg1, arg2):
        result = []
        for (x, y) in zip(arg1, arg2):
            result.append(x + ' ' + y)
        return result

