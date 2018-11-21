import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='object_extractor')
    parser.add_argument('--extractor', dest='extractor')
    parser.add_argument('--data', dest='data')
    parser.add_argument('--output', dest='output')
    args = parser.parse_args()
    return args

args = parse_args()

train_neg_input = args.data + 'training/negative/'
train_pos_input = args.data + 'training/positive/'
test_neg_input = args.data + 'testing/negative/'
test_pos_input = args.data + 'testing/positive/'

train_neg_output = args.output + 'training/negative/'
train_pos_output = args.output + 'training/positive/'
test_neg_output = args.output + 'testing/negative/'
test_pos_output = args.output + 'testing/positive/'

if not os.path.exists(train_neg_output):
    os.makedirs(train_neg_output)

if not os.path.exists(train_pos_output):
    os.makedirs(train_pos_output)

if not os.path.exists(test_neg_output):
    os.makedirs(test_neg_output)

if not os.path.exists(test_pos_output):
    os.makedirs(test_pos_output)

total_videos = len([name for name in os.listdir(train_neg_input)]) + len([name for name in os.listdir(train_pos_input)]) + len([name for name in os.listdir(test_neg_input)]) + len([name for name in os.listdir(test_pos_input)])

print(total_videos)

n = 1

print('Starting negative training set')
for filename in os.listdir(train_neg_input):
    command = args.extractor + ' ' + train_neg_input + filename + ' ' + train_neg_output
    print('Processing video ' + str(n) + '/' + str(total_videos))
    n = n + 1
    os.system(command)

print('Starting positive training set')
for filename in os.listdir(train_pos_input):
    command = args.extractor + ' ' + train_pos_input + filename + ' ' + train_pos_output
    print('Processing video ' + str(n) + '/' + str(total_videos))
    n = n + 1
    os.system(command)

print('Starting negative testing set')
for filename in os.listdir(test_neg_input):
    command = args.extractor + ' ' + test_neg_input + filename + ' ' + test_neg_output
    print('Processing video ' + str(n) + '/' + str(total_videos))
    n = n + 1
    os.system(command)

print('Starting positive testing set')
for filename in os.listdir(test_pos_input):
    command = args.extractor + ' ' + test_pos_input + filename + ' ' + test_pos_output
    print('Processing video ' + str(n) + '/' + str(total_videos))
    n = n + 1
    os.system(command)

