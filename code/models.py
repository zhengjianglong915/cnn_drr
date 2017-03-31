#!/usr/bin/env python
# encoding: utf-8
# 访问 http://tool.lu/pyc/ 查看更多信息
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Embedding, Merge, MaxPooling1D, Convolution1D, Flatten, Bidirectional,GlobalAveragePooling1D,GRU 


def word2Vec_mlp(wordVec, embedding_size, wordDim, maxlen):
    ## 词向量知识表示模型
    ## 固定输入词向量的长度
    left_branch = Sequential()
    left_branch.add(Embedding(embedding_size, output_dim = wordDim, input_length = maxlen, weights = [
        wordVec], mask_zero = False, name = 'embeddings1'))
    right_branch = Sequential()
    right_branch.add(Embedding(embedding_size, output_dim = wordDim, input_length = maxlen, weights = [
        wordVec], mask_zero = False, name = 'embeddings2'))
    merged = Merge([left_branch, right_branch], mode='concat')
    model = Sequential()
    model.add(merged)
    model.add(Flatten())
    model.add(Dense(128, activation = 'sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def bag_of_words_mlp(wordVec, embedding_size, wordDim, maxlen):
    # Bag-of-Words
    left_branch = Sequential()
    left_branch.add(Embedding(embedding_size, output_dim = wordDim, input_length = maxlen, weights = [
        wordVec], mask_zero = False))
    left_branch.add(GlobalAveragePooling1D())
    left_branch.add(Dropout(0.5))

    right_branch = Sequential()
    right_branch.add(Embedding(embedding_size, output_dim = wordDim, input_length = maxlen, weights = [
        wordVec], mask_zero = False))
    right_branch.add(GlobalAveragePooling1D())
    right_branch.add(Dropout(0.5))

    merged = Merge([left_branch, right_branch], mode = 'concat')
    model = Sequential()
    model.add(merged)
    model.add(Dropout(0.5))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))
    return model



def lstm_mlp(wordVec, embedding_size, wordDim, maxlen):
    # lstm + mlp
    left_branch = Sequential()
    left_branch.add(Embedding(embedding_size, output_dim = wordDim, input_length = maxlen, weights = [
        wordVec], mask_zero = False))
    left_branch.add(LSTM(50, return_sequences = True))
    left_branch.add(Dropout(0.5))

    right_branch = Sequential()
    right_branch.add(Embedding(embedding_size, output_dim = wordDim, input_length = maxlen, weights = [
        wordVec], mask_zero = False))
    right_branch.add(LSTM(50, return_sequences = True))
    right_branch.add(Dropout(0.5))

    merged = Merge([left_branch, right_branch], mode = 'concat')
    model = Sequential()
    model.add(merged)
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))
    return model



def bilstm_mlp(wordVec, embedding_size, wordDim, maxlen):
    # bilstm + mlp
    left_branch = Sequential()
    left_branch.add(Embedding(embedding_size, output_dim = wordDim, input_length = maxlen, weights = [
        wordVec], mask_zero = False))
    left_branch.add(Bidirectional(LSTM(50, return_sequences = False)))
    left_branch.add(Dropout(0.5))

    right_branch = Sequential()
    right_branch.add(Embedding(embedding_size, output_dim = wordDim, input_length = maxlen, weights = [
        wordVec], mask_zero = False))
    right_branch.add(Bidirectional(LSTM(50, return_sequences = False)))
    right_branch.add(Dropout(0.5))

    merged = Merge([left_branch, right_branch], mode = 'concat')
    model = Sequential()
    model.add(merged)
    #model.add(Flatten())
    #model.add(Dropout(0.5))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))
    return model


def gru_mlp(wordVec, embedding_size, wordDim, maxlen):
    # gru + mlp
    left_branch = Sequential()
    left_branch.add(Embedding(embedding_size, output_dim = wordDim, input_length = maxlen, weights = [
        wordVec], mask_zero = False))
    left_branch.add(GRU(50, return_sequences = False))
    left_branch.add(Dropout(0.5))

    right_branch = Sequential()
    right_branch.add(Embedding(embedding_size, output_dim = wordDim, input_length = maxlen, weights = [
        wordVec], mask_zero = False))
    right_branch.add(GRU(50, return_sequences = False))
    right_branch.add(Dropout(0.5))

    merged = Merge([left_branch, right_branch], mode = 'concat')
    model = Sequential()
    model.add(merged)
    #model.add(Flatten())
    #model.add(Dropout(0.5))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))
    return model

def bigru_mlp(wordVec, embedding_size, wordDim, maxlen):
    # bi-gru + mlp
    left_branch = Sequential()
    left_branch.add(Embedding(embedding_size, output_dim = wordDim, input_length = maxlen, weights = [
        wordVec], mask_zero = False))
    left_branch.add(Bidirectional(GRU(50, return_sequences = False)))
    left_branch.add(Dropout(0.5))

    right_branch = Sequential()
    right_branch.add(Embedding(embedding_size, output_dim = wordDim, input_length = maxlen, weights = [
        wordVec], mask_zero = False))
    right_branch.add(Bidirectional(GRU(50, return_sequences = False)))
    right_branch.add(Dropout(0.5))

    merged = Merge([left_branch, right_branch], mode = 'concat')
    model = Sequential()
    model.add(merged)
    #model.add(Flatten())
    #model.add(Dropout(0.5))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))
    return model

def gru_cnn_mlp(wordVec, embedding_size, wordDim, maxlen):
    # gru + cnn + mlp
    nb_filter = 50
    filter_length = 5
    pool_length = 5
    top_pool_length = 5
    left_branch = Sequential()
    left_branch.add(Embedding(embedding_size, output_dim = wordDim, input_length = maxlen, weights = [
        wordVec], mask_zero = False))
    left_branch.add(Bidirectional(GRU(50, return_sequences = True)))
    left_branch.add(Dropout(0.5))
    left_branch.add(Convolution1D(nb_filter = nb_filter, filter_length = filter_length, border_mode = 'full', activation = 'relu', subsample_length = 1))
    left_branch.add(MaxPooling1D(pool_length = pool_length))
    left_branch.add(Dropout(0.5))

    right_branch = Sequential()
    right_branch.add(Embedding(embedding_size, output_dim = wordDim, input_length = maxlen, weights = [
        wordVec], mask_zero = False))
    right_branch.add(Bidirectional(GRU(50, return_sequences = True)))
    right_branch.add(Dropout(0.5))
    right_branch.add(Convolution1D(nb_filter = nb_filter, filter_length = filter_length, border_mode = 'full', activation = 'relu', subsample_length = 1))
    right_branch.add(MaxPooling1D(pool_length = pool_length))
    right_branch.add(Dropout(0.5))

    merged = Merge([left_branch, right_branch], mode = 'concat')
    model = Sequential()
    model.add(merged)
    model.add(Convolution1D(nb_filter = nb_filter, filter_length = 2, border_mode = 'full', activation = 'relu', subsample_length = 1))
    model.add(MaxPooling1D(pool_length = top_pool_length))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))
    return model


def lstm_cnn_mlp(wordVec, embedding_size, wordDim, maxlen):
    nb_filter = 50
    filter_length = 3
    pool_length = 5
    top_pool_length = 10
    left_branch = Sequential()
    left_branch.add(Embedding(embedding_size, output_dim = wordDim, input_length = maxlen, weights = [
        wordVec], mask_zero = False))
    left_branch.add(Bidirectional(LSTM(50, return_sequences = True)))
    left_branch.add(Dropout(0.5))
    left_branch.add(Convolution1D(nb_filter = nb_filter, filter_length = filter_length, border_mode = 'full', activation = 'relu', subsample_length = 1))
    left_branch.add(MaxPooling1D(pool_length = pool_length))
    left_branch.add(Dropout(0.5))   

    right_branch = Sequential()
    right_branch.add(Embedding(embedding_size, output_dim = wordDim, input_length = maxlen, weights = [
        wordVec], mask_zero = False))
    right_branch.add(Bidirectional(LSTM(50, return_sequences = True)))
    right_branch.add(Dropout(0.5))
    right_branch.add(Convolution1D(nb_filter = nb_filter, filter_length = filter_length, border_mode = 'full', activation = 'relu', subsample_length = 1))
    right_branch.add(MaxPooling1D(pool_length = pool_length))
    right_branch.add(Dropout(0.5))

    merged = Merge([left_branch, right_branch], mode = 'concat')
    model = Sequential()
    model.add(merged)
    model.add(Convolution1D(nb_filter = nb_filter, filter_length = filter_length, border_mode = 'full', activation = 'relu', subsample_length = 1))
    model.add(MaxPooling1D(pool_length = top_pool_length))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))
    return model
    return model
