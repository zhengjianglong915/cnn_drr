from __future__ import print_function
import copy 
import optparse
import numpy as np
np.random.seed(1337) # for reproducibility
from evaluate import dr_evaluate
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.optimizers import Adagrad
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, Bidirectional
from keras.layers import LSTM, Convolution1D, GlobalMaxPooling1D, Merge
from grn import GRN
from dataUtil import DataUtil
from myMerge import MyMerge
import util as ut
import models as models

## modelName 
#modelName = 'lstm_cnn_mlp'
#modelName = 'lstm_grn_cnn_mlp'
modelName = 'word2Vec_mlp'
#modelName = 'bag_of_words_mlp'
#modelName = 'lstm_mlp'
#modelName = 'bilstm_mlp'
#modelName = 'gru_mlp'
#modelName = 'bigru_mlp'
#modelName = 'gru_cnn_mlp'


## wordVec
wordVec = 'glove'
#wordVec = 'word2Vec'

## rel
#rel = "comparison"
#rel = "temporal"
#rel = "expansion"
rel = "contingency"
## Embedding

if wordVec == 'glove':
    embedding_size = 201319
else:
    embedding_size = 400001  #word2Vec

wordDim = 50
maxlen = 50

# Training
batch_size = 32
nb_epoch = 25

#paramter="filter50_filterLength3_poolLength10_toppoollength10"
paramter=wordVec + "_" + str(wordDim) 

def run(args=None):
    maxF_dev = 0 
    bestTestF_dev = 0
    bestTestAcc_dev = 0
    allBestTestF_dev = 0
    allBestTestAcc_dev = 0
    avgBestTestF_dev = 0
    avgBestTestAcc_dev =0
    for i in range(100):
        devBestTestF, bestTestAcc = train_and_test(modelName, embedding_size, wordDim, maxlen, wordVec)
        
        allBestTestF_dev = allBestTestF_dev + devBestTestF
        allBestTestAcc_dev = allBestTestAcc_dev + bestTestAcc

        l = i + 1
        avgBestTestF_dev = allBestTestF_dev / l
        avgBestTestAcc_dev = allBestTestAcc_dev / l 

        if bestTestF_dev < devBestTestF:
            bestTestF_dev = devBestTestF
            bestTestAcc_dev = bestTestAcc  

        output=open("../result/"+ modelName + "_" + paramter +"_" + rel +".txt", "a")
        output.write("model: "+modelName + "  paramter:" + paramter +"\r\n")
        output.write( str(i)+ "## " +rel + " vs others.\r\n")
        output.write(" --devBestTestF:"+ str(devBestTestF)+ "  maxF:"+str(bestTestF_dev)+"\r\n")
        output.write(" --bestTestAcc:" + str(bestTestAcc)+ " maxAcc:"+str(bestTestAcc_dev)+"\r\n")
        output.write(" --avgBestTestF_dev::" + str(avgBestTestF_dev)+ " avgBestTestAcc_dev:"+str(avgBestTestAcc_dev)+"\r\n")
        output.close()    

    output=open("../result/"+ modelName + "_" + paramter +"_" + rel +".txt", "a")
    output.write("model: "+modelName + "  paramter:" + paramter  +"\r\n")
    output.write(" ------------------------------------\r\n")
    output.write("best f_score:"+str(bestF) + "\r\n")
    output.write("best acc:"+str(bestAcc) + "\r\n")
    output.close()


def train_and_test(modleName, embedding_size, wordDim, maxlen, wordVec):
    print("loading data...")
    dataUtil = DataUtil()
    wordMap, wordVec, train_arg1,train_arg2,train_label,dev_arg1,dev_arg2, dev_label,test_arg1, test_arg2, test_label = dataUtil.loadData(rel, wordDim, wordVec)
    print(len(train_arg1), 'train sequences')
    print(len(dev_arg1), "dev sequences")
    print(len(test_arg1), 'test sequences')
    print('Pad sequences(samples x time)')
    train_arg1 = sequence.pad_sequences(train_arg1, maxlen=maxlen,  padding='post', truncating='post')
    train_arg2 = sequence.pad_sequences(train_arg2, maxlen=maxlen, padding='post', truncating='post')
    test_arg1 = sequence.pad_sequences(test_arg1, maxlen=maxlen, padding='post', truncating='post')
    test_arg2 = sequence.pad_sequences(test_arg2, maxlen=maxlen, padding='post', truncating='post')
    dev_arg1 = sequence.pad_sequences(dev_arg1, maxlen=maxlen, padding='post', truncating='post')
    dev_arg2 = sequence.pad_sequences(dev_arg2, maxlen=maxlen, padding='post', truncating='post')
   
    print('Build model...')
    model = buildBranchModel(modelName, wordVec,embedding_size, wordDim, maxlen)
    adam = Adagrad(lr=0.01, epsilon=1e-06)
    model.compile(loss='binary_crossentropy',
              metrics=[ut.f_score], optimizer=adam)
    
    print("use modle:", modelName) 
    print(rel + " vs ohters.")
    print('Train...')
    bestDevF = 0
    devBestTestF =0
    devBestTestAcc = 0
    bestTestF = 0
    bestTestAcc = 0 
    dataUtil = DataUtil()
    for each in range(nb_epoch):
        model.fit([train_arg1, train_arg2],train_label, batch_size = batch_size, nb_epoch=1, validation_data=([dev_arg1, dev_arg2], dev_label))
        devResult = model.predict_classes([dev_arg1, dev_arg2], batch_size=batch_size, verbose=1)
        df_measure, dpre, drecall, dacc =dr_evaluate(dev_label, devResult)
        print("["+str(each) + '] dev F-measure:' + str(df_measure) +"  dev acc:" + str(dacc))
        result = model.predict_classes([test_arg1, test_arg2], batch_size=batch_size, verbose=1)
        f_measure, pre, recall, acc =dr_evaluate(test_label, result)
        if f_measure > bestTestF:
            bestTestF = f_measure
            bestTestAcc = acc         
        if bestDevF < df_measure:        
            bestDevF = df_measure
            devBestTestF = f_measure
            devBestTestAcc = acc

        print("test f:" + str(f_measure) + " acc:" + str(acc) + "  bestF:" + str(bestTestF) + "  bestAcc:"+str(bestTestAcc) + " devBestTestF:" + str(devBestTestF) + " devBestTestAcc:" + str(devBestTestAcc)) 
        (train_arg1, train_arg2, train_label) = dataUtil.load_traindata(rel, wordMap)
        train_arg1 = sequence.pad_sequences(train_arg1, maxlen=maxlen,  padding='post', truncating='post')
        train_arg2 = sequence.pad_sequences(train_arg2, maxlen=maxlen, padding='post', truncating='post')
                
    print("**********************************************************************")
    print("**********************************************************************")
    print("use modle:", modelName + " mode:" + mode)
    print(rel + " vs others.")
    print("dev bestF:" + str(bestDevF) + " testF:" + str(devBestTestF) + " testAcc:" + str(devBestTestAcc))
    print("best test: f_score " + str(bestTestF) +"  bestTestAcc " + str(bestTestAcc))
    print("**********************************************************************")
    print("**********************************************************************")
    return  devBestTestF, devBestTestAcc


def buildBranchModel(modelName, wordVec, embedding_size, wordDim, maxlen):
    if modelName == 'bag_of_words_mlp':
         model = models.bag_of_words_mlp(wordVec, embedding_size, wordDim, maxlen)    
    elif modelName == 'word2Vec_mlp':
        model = models.word2Vec_mlp(wordVec, embedding_size, wordDim, maxlen) 
    elif modelName == 'lstm_mlp':
        model = models.lstm_mlp(wordVec, embedding_size, wordDim, maxlen)
    elif modelName == 'bilstm_mlp':
        model = models.bilstm_mlp(wordVec, embedding_size, wordDim, maxlen)
    elif modelName == 'gru_mlp':
        model = models.gru_mlp(wordVec, embedding_size, wordDim, maxlen)
    elif modelName == 'bigru_mlp':
        model = models.bigru_mlp(wordVec, embedding_size, wordDim, maxlen)
    elif modelName == 'gru_cnn_mlp':
        model = models.gru_cnn_mlp(wordVec, embedding_size, wordDim, maxlen)
    elif modelName == 'lstm_cnn_mlp':
        model = models.lstm_cnn_mlp(wordVec, embedding_size, wordDim, maxlen)
    return model
    
def buildModel(modelName, wordVec, embedding_size, wordDim, maxlen):
    if modelName == 'LSTM_MLP':
        model = lstm_mlp.buildModel(wordVec, embedding_size, wordDim, maxlen)
   
    return model

if __name__ == '__main__':
    run()


