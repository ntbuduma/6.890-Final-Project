
import numpy as np
from bitarray import bitarray
import random
import pandas as pd
import csv
import hashlib

df = pd.read_csv("final_dataset2.csv")
email_list = list(df.url)
label_list = list(df.label)
malicious = {}
good = {}
b1 = 10

for index, row in df.iterrows():
    if row.label == 1:
        malicious[row.url] = row.label
    elif row.label == -1:
        good[row.url] = row.label

bloom_filter = [False for i in range(b1*len(malicious))]

for bad in malicious:
    index = hash(bad) % (b1*len(malicious))
    bloom_filter[index] = True

count = 0
for url, label in good.items():
    index = hash(url) % (b1*len(malicious))
    is_in = bloom_filter[index]
    if is_in:
        count += 1

print("false positive rate: " + str(count/len(good)))
