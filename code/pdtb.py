import nltk
import sys  
import numpy as np
from loadWordVec import load_wordvec

reload(sys)  
sys.setdefaultencoding('utf8')

def top10000_words():
     wordMap = {}
     static_num("train_arg1.txt", wordMap)
     static_num("train_arg2.txt", wordMap)
     wordMapList = sorted(wordMap.iteritems(), key=lambda d:d[1], reverse = True)         
     
     (wordVec, wordMap) = load_wordvec('vectors/turian50.txt', 50)    
     newWordMap = {}
     count = 0
     for (word, val) in wordMapList:
         if count == 10000:
             break
         if word in wordMap:
             newWordMap[word] = wordVec[wordMap[word]]
             count += 1
         else:
             print word, str(val)
                     
     
     newWord2Vec = open("vectors/newWord2Vec.txt", 'w')
     newWord2Vec.write("UNK "+  list2str(wordVec[0]) + "\n")
     for (key, value) in newWordMap.items():
         newWord2Vec.write(key+" "+ list2str(value) + "\n")
     newWord2Vec.close()     

def list2str(lis):
    result = ""
    for val in lis:
        result = result + str(val) + " "
    return result

def static_num(fileName, wordMap):
    data = open("data/"+ fileName, 'r')
    for line in data:
        line = line.replace('\r\n', '')
        words = nltk.word_tokenize(line)
        #words = [word.lower() for word in words]
        for word in words:
            #word = word.lower()
            if word in wordMap:
                wordMap[word] += 1
            else:
                wordMap[word] = 1
    data.close()



top10000_words()
