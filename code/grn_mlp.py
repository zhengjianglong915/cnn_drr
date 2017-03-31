from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Embedding, Merge,Bidirectional, MaxPooling1D, Flatten
from myMerge import MyMerge

# LSTM + MLP

def buildBranchModel(wordVec, embedding_size, wordDim, maxlen):
    
    left_branch = Sequential()
    left_branch.add(Embedding(embedding_size, output_dim=wordDim,
                     input_length=maxlen,  weights=[wordVec], mask_zero=False, name="embeddings"))
    left_branch.add(Bidirectional(LSTM(50, return_sequences = True)))
    #left_branch.add(MaxPooling1D(pool_length=10))
    #left_branch.add(Flatten())
   

    right_branch = Sequential()
    right_branch.add(Embedding(embedding_size, output_dim=wordDim,
                     input_length=maxlen,  weights=[wordVec], mask_zero=False, name="embeddings"))
    right_branch.add(Bidirectional(LSTM(50, return_sequences = True)))
    #right_branch.add(MaxPooling1D(pool_length=10))
    #right_branch.add(Flatten())
      

    merged = MyMerge([left_branch, right_branch], slic = 3)
    model = Sequential()
    model.add(merged)
    model.add(MaxPooling1D(pool_length=10))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def buildModel(wordVec, embedding_size, wordDim, maxlen):
    model = Sequential()
    model.add(Embedding(embedding_size, output_dim=wordDim,
                     input_length=maxlen,  weights=[wordVec], mask_zero=True))
    model.add(LSTM(128))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model
