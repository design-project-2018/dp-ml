import pandas as pd
import os
import random
import math
import json

# function to round a number up or down to nearest int with 0.5 probability
def random_round(num):
    prob = random.uniform(0, 1)
    return math.ceil(num) if prob > 0.5 else math.floor(num)

# load dataset and aggregate scores both different labellers
labels = pd.read_csv('labels/cleaned.csv', header=None)
labels = labels.groupby([1, 2])[3].mean().reset_index(name=3)
for index, label in labels.iterrows():
    labels.loc[index, 3] = random_round(label[3])

# walk through all json files and add labels
for root, directories, filenames in os.walk('object_extraction'):
    for filename in filenames:   
        print('Processing file: {}'.format(filename))
        with open(os.path.join(root,filename), 'r+') as f:
            data = json.load(f)
            data['label'] = int(labels.loc[labels[2] == filename.replace("json", "mp4")].reset_index().loc[0,3])
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=4)





