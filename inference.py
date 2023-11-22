from transformers import AutoModelForTokenClassification, pipeline, BertTokenizer
import pandas as pd, numpy as np

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

path_to_file = input('Enter path to the text with names of Indian mountains')

with open(path_to_file, "r") as f: # take a text with indian mountains
    data = f.read()

results = classifier(data) # take a results
location_filter = [results[i] for i in range(len(results)) 
                   if results[i]['entity'] == 'B-LOC' or results[i]['entity'] == 'I-LOC']
finded_mountains = [location_filter[i] for i in range(len(location_filter)) if location_filter[i]['word'] in mountains]

with open(path_to_file, "r") as f: # write locations of finded mountains
    f.write(finded_mountains)