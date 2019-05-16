import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from rnn import lineToTensor, categoryFromOutput, RNN

df = pd.read_csv("final_dataset2.csv")
malicious = {}
good = {}

for index, row in df.iterrows():
    if row.label == 1:
        malicious[row.url] = row.label
    elif row.label == -1:
        good[row.url] = row.label

# early bloom filter
b1 = 4
early_bloom_filter = [False for i in range(b1*len(malicious))]

for bad in malicious:
    index = hash(bad) % (b1*len(malicious))
    early_bloom_filter[index] = True

# count = 0
# first_output = []
# for url, label in good.items():
#     index = hash(url) % (b1*len(malicious))
#     is_in = bloom_filter[index]
#     if is_in:
#         count += 1
#         first_output.append(url)

# rnn = torch.load("rnn.latest")
# rnn.eval()

# count = 0
# false_negatives = []
# for url in first_output:
    # hidden = rnn.initHidden()
    # tensor_url = lineToTensor(url)
#     # predict_url, predict_cat = drawRandomTrainingExample()
#     for i in range(tensor_url.size()[0]):
#         output, hidden = rnn(tensor_url[i], hidden)
#     if categoryFromOutput(output) == 0:
#         count += 1
#         false_negatives.append(url)
    # output_cat = categoryFromOutput(output)
    # print("output: " + str(output_cat))

rnn = torch.load("rnn.latestexp4")
rnn.eval()

print("Hi")

# second bloom filter
b2 = 6
# print("false negatives rate: " + str(count/len(malicious)))
bloom_filter = [False for i in range(b2*len(malicious))]
for bad in malicious:
    hidden = rnn.initHidden()
    tensor_url = lineToTensor(bad)
    # predict_url, predict_cat = drawRandomTrainingExample()
    for i in range(tensor_url.size()[0]):
        output, hidden = rnn(tensor_url[i], hidden)
    # catches false negatives 
    if categoryFromOutput(output) == 0:
        index = hash(bad) % (b2*len(malicious))
        bloom_filter[index] = True

# count = 0
# for url in false_negatives:
#     index = hash(url) % (b2*len(malicious))
#     is_in = bloom_filter[index]
#     if is_in:
#         count += 1

# print("false positive rate: " + str(count/len(false_negatives)))
false_positives = 0
for url in good:
    index = hash(url) % (b1*len(malicious))
    if early_bloom_filter[index]:
        hidden = rnn.initHidden()
        tensor_url = lineToTensor(url)
        for i in range(tensor_url.size()[0]):
            output, hidden = rnn(tensor_url[i], hidden)
        if categoryFromOutput(output) == 1:
            false_positives += 1
            continue
        else:
            index = hash(url) % (b2*len(malicious))
            if bloom_filter[index]:
                false_positives += 1
print("false positive rate: " + str(false_positives/len(good.keys())))
