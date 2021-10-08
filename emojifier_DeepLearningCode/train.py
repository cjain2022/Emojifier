import numpy as np 
import pandas as pd
import emoji as emoji 

# Preparing Dataset
train=pd.read_csv('dataset/train_emoji.csv',header=None)
test=pd.read_csv('dataset/test_emoji.csv',header=None)

print(train.head())
# - collumn 0 :-> stores the textual data which was input by user
# - collumn 1 :-> stores the id for the emoji which was predicted for the input text
# - Other collumns :-> can be ignored

# Mapping id of emoji to text_code_for_emoji 
emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":beaming_face_with_smiling_eyes:",
                    "3": ":downcast_face_with_sweat:",
                    "4": ":fork_and_knife:",
                   }

# We are trying to convert the text_code fork_and_knife into its corresponding graphical emoji using emoji.emojize
print(emoji.emojize(":fork_and_knife:"))

# Print Emojis We Have in Our Dictionary
for e in emoji_dictionary.values():
    print(emoji.emojize(e))

#  We have 5 Emojis For Prediction, so we will convert output to categorical output of shape ( , 5)
from tensorflow.keras.utils import to_categorical 
XT = train[0]
Xt = test[0]

YT = to_categorical(train[1])
Yt = to_categorical(test[1])


print("Dimension of training data",XT.shape)
print("Dimension of testing data",Xt.shape)
print("Dimension of Ytrain after using to_categorical",YT.shape)
print("Dimension of Ytest after using to_categorical",Yt.shape)

# Loading Glove Vector As Dictionary 
embeddings = {}
with open('embedding/glove.6B.50d.txt',encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:],dtype='float32')
        
        #print(word)
        #print(coeffs)
        embeddings[word] = coeffs

# Oberserving Dimension Of Embedding 
print("Testing embedding shapes")
print("Size of dictionary:->",len(embeddings))
print("Length of embedding for one word 'the'is  :->",embeddings['the'].shape)

# Observing Maximum Words in Input String
max_words=0
for sent in XT:
    max_words=max(max_words,len(sent.split(' ')))
print("Max Word in Input string are =",max_words)

# Function To Generate Embedding For Input Strings 
def getOutputEmbeddings(X):
    
    embedding_matrix_output = np.zeros((X.shape[0],10,50))
    for ix in range(X.shape[0]):
        X[ix] = X[ix].split()
        for jx in range(len(X[ix])):
            embedding_matrix_output[ix][jx] = embeddings[X[ix][jx].lower()]
            
    return embedding_matrix_output

emb_XT = getOutputEmbeddings(XT)
emb_Xt = getOutputEmbeddings(Xt)
print("Shape of Embeddings of XT",emb_XT.shape)
print("Shape of Embeddings of Xt",emb_Xt.shape)

print("Train Data Present= ",XT[1])
print("Embedding For Above Text of XT[1]= ",emb_XT[1])

# Deep Learning Model To Map Input TO Output 
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.models import Sequential
model = Sequential()
model.add(LSTM(64,input_shape=(10,50),return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(64,input_shape=(10,50)))
model.add(Dropout(0.3))
model.add(Dense(5))
model.add(Activation('softmax'))
print(model.summary())

