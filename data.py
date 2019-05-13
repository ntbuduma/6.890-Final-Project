import sys
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt

one = pd.read_csv('./Dataset/data.csv', usecols=['url', 'label'])
two = pd.read_csv('./Dataset/Malicious URL.csv', usecols=['url'])

bad_count = 0
good_count = 0
with open('final_dataset.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['url', 'label'])
    for index, row in two.iterrows():
    	writer.writerow([row.url, +1])
    	bad_count += 1
    for index, row in one.iterrows():
    	if row.label == "bad":
    		try:
	    		writer.writerow([row.url, +1])
	    		bad_count += 1
	    	except UnicodeEncodeError:
	    		continue
    		
    	elif row.label == "good":
    		try:
	    		writer.writerow([row.url, -1])
	    		good_count += 1
	    	except UnicodeEncodeError:
	    		continue
    		if good_count * 4 >= bad_count:
    			break

df = pd.read_csv('final_dataset2.csv')
df.drop_duplicates(subset=None, inplace=True)
df.to_csv('final_dataset2.csv')
# print(list(df.url))
