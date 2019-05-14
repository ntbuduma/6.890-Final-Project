import torch
import torch.nn as nn
import pandas as pd
import numpy as np

def fetch_urls_labels():
    data = pd.read_csv("final_dataset2.csv").url.tolist()
    labels = pd.read_csv("final_dataset2.csv").label.tolist()
    return data, labels

data, labels = fetch_urls_labels()
all_letters = list(set(''.join(data)))

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.index(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, len(all_letters))
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

# all_lines = []
# for datum in data:
#     all_lines.append(lineToTensor(datum))

# all_labels = []
# for label in labels:
#     if label == -1:
#         label += 1
#     all_labels.append(torch.tensor(label,dtype=torch.long))

def drawRandomTrainingExample():
    index = np.random.choice(len(all_lines), 1)[0]
    return all_lines[index], all_labels[index]

# def drawRandomTestExample():
#     test_lines, test_labels = all_lines[int(0.8*len(all_lines)):], all_labels[:int(0.8*len(all_labels)):]
#     index = np.random.choice(len(test_lines), 1)[0]
#     return test_lines[index], test_labels[index]

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


criterion = nn.NLLLoss()
learning_rate = 0.0005

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()
    # print(line_tensor, line_tensor.shape)
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    category_tensor = category_tensor.view(1)
    # print(category_tensor, output)
    # print(output.shape, category_tensor.shape)
    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

def test(input_line, n_predictions=1):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i

if __name__ == "__main__":
    # n_hidden = 128
    # rnn = RNN(len(all_letters), n_hidden, 2)
    # rnn.train()
    # iterations = 100000
    # print_every = 1000
    # print("Starting training...")
    # for i in range(iterations):
    #     #print("iteration: ", i)
    #     url, category= drawRandomTrainingExample()
    #     output, loss = train(category, url)

    #     # Print iter number, loss, name and guess
    #     if i % print_every == 0:
    #         print('%d %d%% %.4f' % (i, i / iterations * 100, loss))
    
    # torch.save(rnn, "rnn.latest")
    list_b = [(a, 10-a) for a in range(1, 10)]
    # print(list_b)
    for b1, b2 in list_b:
        df = pd.read_csv("final_dataset2.csv")
        malicious = {}
        good = {}

        for index, row in df.iterrows():
            if row.label == 1:
                malicious[row.url] = row.label
            elif row.label == -1:
                good[row.url] = row.label

        # early bloom filter
        
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

        rnn = torch.load("rnn.latest")
        rnn.eval()

        # second bloom filter
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
        print(str(b1) + ", " + str(b2) + " false positive rate: " + str(false_positives/len(good.keys())))






        
