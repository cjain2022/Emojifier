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