from emoji import emojize
import numpy as np
import pandas as pd
from keras.models import model_from_json
from pathlib import Path
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
DEEP_LEARNING_BASE_DIR = Path(__file__).resolve().parent
import os 
print("-- Loading Model File")
model_json_file = os.path.join(DEEP_LEARNING_BASE_DIR,'trainedModel/model.json')
with open(model_json_file, "r") as file:
    model = model_from_json(file.read())

model_weight_file = os.path.join(DEEP_LEARNING_BASE_DIR,'trainedModel/model.h5')
model.load_weights(model_weight_file)

print("-- Loading Embeddings File")
embeddings = {}
glove_path=os.path.join(DEEP_LEARNING_BASE_DIR,'embedding/glove.6B.50d.txt')
with open(glove_path,encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:],dtype='float32')
        embeddings[word] = coeffs

def getOutputEmbeddings(X):
    
    embedding_matrix_output = np.zeros((X.shape[0],10,50))
    for ix in range(X.shape[0]):
        X[ix] = X[ix].split()
        for jx in range(len(X[ix])):
            if embeddings.get(X[ix][jx].lower()) is not None:
                embedding_matrix_output[ix][jx] = embeddings[X[ix][jx].lower()]
            
    return embedding_matrix_output

def predict(input_str):
    emoji_dictionary = {
        "0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
        "1": ":baseball:",
        "2": ":beaming_face_with_smiling_eyes:",
        "3": ":downcast_face_with_sweat:",
        "4": ":fork_and_knife:",
    }

    X = pd.Series([input_str])
    emb_X = getOutputEmbeddings(X)
    p=np.argmax(model.predict(emb_X), axis=-1)

    return emojize(emoji_dictionary[str(p[0])])
    
if __name__ == "__main__":
    test_str = "i love it"
    output = predict(test_str)
    print(emojize(output))

