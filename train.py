from transformers import AutoModelForTokenClassification, pipeline, BertTokenizer
import pandas as pd, numpy as np
import os


tokenizer = BertTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

with open("names_of_mountains.txt", "r") as f: # take names of mountains
    mountains = f.read()

# # if we need a more accuracy: 
# mountains2 = []
# for mount in mountains:
#     temp = mount.split() # divide by space
#     if len(temp) > 1:
#         for word in temp:
#         # print(word)
#           if len(word) > 2: # only big words append to list
#               mountains2.append(word)
#     else:
#         mountains2.extend(temp)
# mountains.extend(mountains2)

tokenizer.add_tokens(mountains) # update the tokenizer
model.resize_token_embeddings(len(tokenizer))

classifier = pipeline('ner', model = model, tokenizer = tokenizer)

PATH = '/train' # take a path to files

total_results = dict()
for root, dirs, files in os.walk(PATH):
    for filename in files:
        # print(filename)
        with open(PATH + '/train/' + filename, "r", encoding = 'utf-8') as f:
            data = f.read() # for each text
            results = classifier(data) # take a result
            # filter only location
            filter1 = [results[i] for i in range(len(results)) if results[i]['entity'] == 'B-LOC' or results[i]['entity'] == 'I-LOC']
            # take names of mountains from the location list
            mounts = [filter1[i] for i in range(len(filter1)) if filter1[i]['word'] in mountains]
            total_results[filename] = mounts # add to dictionary new data

res = pd.DataFrame()
res['document'] = list(total_results.keys()) # names of files
res['len_of_searched_names'] = [len(total_results[key]) for key in total_results.keys()] # number of searched names of mountains