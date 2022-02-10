#!/usr/bin/env python

import os
import csv
import copy
import random
import operator
import re
from ast import literal_eval
import numpy as np
import pprint

# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Global variables
TRAINING_DATA_FP = './gene-trainF18.txt'
TEST_SET_FP = './F18-assgn4-test.txt'
OUTPUT_FP = './prendergast-daniel-assgn4-test-output.txt'
GOLD_OUTPUT_FP = './gold_test.txt'
TRAIN_TEST_SPLIT = 0.90
FIND_UNK_DATA_SPLIT = 0.95
SIMPLE_WORDSHAPE = True

def main():
    # read in the training data
    data_raw = read_csv(TRAINING_DATA_FP)
    obs = None
    if TEST_SET_FP!=TRAINING_DATA_FP:
        obs = read_csv(TEST_SET_FP)
    else:
        data_raw, obs = split_data(data_raw, TRAIN_TEST_SPLIT)
        write_results(obs, GOLD_OUTPUT_FP)
    data_with_spaces = encode_spaces(data_raw)
    data_processed = encode_shapes(data_with_spaces)
    obs_with_spaces = encode_spaces(obs)

    # get a list of all the iob tags, word shapes, and types
    print "Getting lists of tags, types, and shapes ..."
    iob_list = get_list(data_processed, 2, add_unknown=False)
    type_list = get_list(data_processed, 1, add_unknown=True)
    shape_list = get_list(data_processed, 3, add_unknown=True)
    # print "IOB List: ", iob_list
    # print "Shape List: ", shape_list

    # get the transition counts, smooth them, and calculate probabilities
    print "Getting transition probs ..."
    transition_counts = get_transition_counts(data_processed, iob_list)
    trans_counts_smooth = add_one_smoothing(transition_counts, exceptions=[('O', 'I'), ('<blank>', 'I')])
    trans_probs = calc_probs(trans_counts_smooth)
    # for key in trans_probs:
    #     print key
    #     print "     ", trans_probs[key]

    # get the observation counts of types given tags and calculate probabilities
    print "Getting type obs probabilities ..."
    type_obs_counts = get_obs_counts(data_processed, iob_list, type_list, tag_index=2, feature_index=1)
    type_obs_counts = add_one_smoothing(type_obs_counts, exceptions=[])
    unk_type_adjustment = find_unk_percents(data_processed, iob_list, FIND_UNK_DATA_SPLIT, tag_index=2, feature_index=1)
    type_obs_counts = adjust_obs_for_unk_smart(type_obs_counts, unk_type_adjustment)
    type_obs_probs = calc_probs(type_obs_counts)

    # get the observation counts of shapes given tags and calculate probabilities
    print "Getting shape obs probabilities ..."
    shape_obs_counts = get_obs_counts(data_processed, iob_list, shape_list, tag_index=2, feature_index=3)
    shape_obs_counts = add_one_smoothing(shape_obs_counts, exceptions=[])
    unk_shape_adjustment = find_unk_percents(data_processed, iob_list, FIND_UNK_DATA_SPLIT, tag_index=2, feature_index=3)
    shape_obs_counts = adjust_obs_for_unk_smart(shape_obs_counts, unk_shape_adjustment)
    shape_obs_probs = calc_probs(shape_obs_counts)

    # print "Unk Types..."
    # for key in unk_type_adjustment:
    #     print key
    #     print "     ", unk_type_adjustment[key]
    # print "Unk Shapes..."
    # for key in unk_shape_adjustment:
    #     print key
    #     print "     ", unk_shape_adjustment[key]

    # calculate pos priors
    print "Calculating IOB priors ..."
    iob_priors = get_priors(data_processed, iob_list, tag_index=2)
    # calculate best state sequence and best probability of that state sequence
    print "Beginning Viterbi ..."
    best_seq = viterbi(obs_with_spaces, iob_list, [type_list, shape_list], trans_probs, [type_obs_probs, shape_obs_probs], iob_priors)

    # create the final output array and write to file
    output_array = []
    for indx, row in enumerate(obs):
        output_row = []
        if row:
            output_row = [row[0], row[1], best_seq[indx]]
        output_array.append(output_row)
    print "Verify output length matches input: ", len(best_seq), len(output_array), len(obs)
    write_results(output_array, OUTPUT_FP)

def viterbi(obs, states, feature_lists, a, feature_obs, priors):


    type_list = feature_lists[0]
    shape_list = feature_lists[1]
    b = feature_obs[0]
    c = feature_obs[1]

    sentences = tokenize_sentences(obs)
    total_seq = []
    for sent_indx, sent in enumerate(sentences):
        num_obs = len(sent)
        num_states = len(states)
        v = np.zeros(shape=(num_states, num_obs))
        backptr = np.zeros(shape=(num_states, num_obs), dtype=int)
        for indx, state in enumerate(states):
            token = sent[0][1]
            if token not in type_list:
                token = '<unknown>'
            token_shape = wordshape(sent[0][1])
            if token_shape not in shape_list:
                token_shape = '<unknown>'
            v[indx][0] = a['<blank>'][state] * priors[state] * b[state][token] * c[state][token_shape]

        for indx in range(num_obs-1):
            token_indx = indx + 1
            token = sent[token_indx][1]
            if token not in type_list:
                token = '<unknown>'
                # print "     Encountered unknown word"
            token_shape = wordshape(sent[token_indx][1])
            if token_shape not in shape_list:
                token_shape = '<unknown>'
                # print "     Encountered unknown shape"
            for indx1, state1 in enumerate(states):
                max_val = 0.0
                max_arg = 0
                for indx2, state2 in enumerate(states):
                    temp_val = v[indx2][token_indx-1] * a[state2][state1] * b[state1][token] * c[state1][token_shape]
                    temp_arg = indx2
                    if temp_val>max_val:
                        max_val = temp_val
                        max_arg = temp_arg
                v[indx1][token_indx] = max_val
                backptr[indx1][token_indx] = max_arg

        # if sent_indx==0:
        #     print "Sentence: ", [row[1] for row in sent]
        #     # pp = pprint.PrettyPrinter(width=160)
        #     # pp.pprint(v)
        #     sent_str = [row[1] for row in sent]
        #     print "%s" % sent_str
        #     for _state, state in enumerate(states):
        #         np.set_printoptions(precision=2, edgeitems=10)
        #         np.core.arrayprint._line_width = 180
        #         print "     ", state, v[_state]
        ptr_indx = np.argmax(v[:, num_obs-1])
        best_seq = []
        for indx in range(num_obs):
            token_indx = num_obs - 1 - indx
            tag = states[ptr_indx]
            best_seq.append(tag)
            temp_ptr_indx = backptr[ptr_indx][token_indx]
            ptr_indx = temp_ptr_indx
        best_seq.reverse()
        best_seq.append('<blank>')
        total_seq = total_seq + best_seq
    return total_seq

def add_one_smoothing(counts, exceptions):
    for key1 in counts:
        for key2 in counts[key1]:
            if (key1, key2) not in exceptions:
                counts[key1][key2] +=1
    return counts

def adjust_obs_for_unk(obs_counts, percent):
    for tag in obs_counts:
        total = 0.0
        for type in obs_counts[tag]:
            total += obs_counts[tag][type]
        obs_counts[tag]['<unknown>'] = total * (percent/100.0)
    return obs_counts

def adjust_obs_for_unk_smart(obs_counts, adjust_dict):
    for tag in obs_counts:
        total = 0.0
        for ftr in obs_counts[tag]:
            total += obs_counts[tag][ftr]
        obs_counts[tag]['<unknown>'] = total * adjust_dict[tag]
    return obs_counts

def calc_probs(counts):
    probs = copy.deepcopy(counts)
    for key1 in counts:
        total = 0.0
        for key2 in counts[key1]:
            total += counts[key1][key2]
        for key2 in counts[key1]:
            if total!=0.0:
                probs[key1][key2] = counts[key1][key2]/total
            else:
                probs[key1][key2] = 1.0/len(counts)
    return probs

def encode_shapes(data):
    data_with_shapes = []
    for row in data:
        shape = wordshape(row[1])
        row.append(shape)
        data_with_shapes.append(row)
    return data_with_shapes

def encode_spaces(data):
    data_with_spaces = []
    for row in data:
        new_row = row
        if not row:
            new_row = ['0', '<blank>', '<blank>']
        data_with_spaces.append(new_row)
    return data_with_spaces

def find_unk_percents(data, tag_list, data_split, tag_index, feature_index):
    known_set = set([])
    num_samples = len(data)
    split_indx = int(round(data_split * num_samples))
    unk_counts = {}
    for tag in tag_list:
        unk_counts[tag] = {}
        unk_counts[tag]['total'] = 0.0
        unk_counts[tag]['num_unknowns'] = 0.0
    for indx in range(num_samples):
        if indx<split_indx:
            known_set.add(data[indx][feature_index])
        else:
            tag = data[indx][tag_index]
            unk_counts[tag]['total'] += 1.0
            if data[indx][feature_index] not in known_set:
                unk_counts[tag]['num_unknowns'] += 1.0
    unk_probs = {}
    for tag in unk_counts:
        unk_probs[tag] = unk_counts[tag]['num_unknowns']/unk_counts[tag]['total']
    return unk_probs

def get_obs_counts(data, tag_list, type_list, tag_index, feature_index):
    obs_counts = {}
    for tag in tag_list:
        obs_counts[tag] = {}
        for type in type_list:
            obs_counts[tag][type] = 0.0
    for row in data:
        type = row[feature_index]
        tag = row[tag_index]
        obs_counts[tag][type] += 1.0
    return obs_counts

def get_list(array, indx, add_unknown):
    lst = []
    for row in array:
        val = row[indx]
        if val not in lst:
            lst.append(val)
    if add_unknown:
        lst.append('<unknown>')
    return lst

def get_priors(data, tag_list, tag_index):
    priors = {}
    for tag in tag_list:
        priors[tag] = 0.0
    total = 0.0
    for row in data:
        if row:
            tag = row[tag_index]
            priors[tag] += 1.0
            total += 1.0
    for tag in priors:
        priors[tag] = priors[tag] / total
    return priors

def get_pos_stats(data):
    word_dict = {}
    pos_dict = {}
    # Collect statistics for each token and its part of speech
    for row in data:
        if row:
            word = row[1]
            pos = row[2]
            if word not in word_dict:
                word_dict[word] = {}
            if pos not in word_dict[word]:
                word_dict[word][pos] = 0
            word_dict[word][row[2]] += 1
            if pos not in pos_dict:
                pos_dict[pos] = 0
            pos_dict[pos] += 1
    return pos_dict, word_dict

def get_transition_counts(data, tag_list):
    transition_counts_dict = {}
    for tag1 in tag_list:
        transition_counts_dict[tag1] = {}
        for tag2 in tag_list:
            transition_counts_dict[tag1][tag2] = 0.0
    prev_tag = '<blank>'
    for row in data:
        current_tag = row[2]
        transition_counts_dict[prev_tag][current_tag] += 1.0
        prev_tag = current_tag
    return transition_counts_dict

def read_csv(fp):
    data = []
    with open(fp,'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            data.append(row)
        csvfile.close()
    return data

def split_data(data, perc):
    split_index = int(round(perc * len(data)))
    train = data[:split_index]
    test = data[split_index:]
    return train, test

def tokenize_sentences(data):
    sentences = []
    sent = []
    num_blanks = 0
    for row in data:
        token = row[1]
        if token!='<blank>':
            sent.append(row)
        else:
            num_blanks += 1
            sentences.append(copy.deepcopy(sent))
            sent = []
    sentences.append(copy.deepcopy(sent))
    total_words = 0
    for sent in sentences:
        total_words += len(sent)
    total_words += num_blanks
    if len(data)!=total_words:
        print "Error -- Words lost during sentence tokenization: ", total_words + num_blanks, len(data)
    return sentences

def train_simple_tagger(pos_histogram, word_pos_histogram):
    # Find the most probable pos for each type
    pos_dict = {}
    for word in word_pos_histogram:
        most_likely_pos = max(word_pos_histogram[word].iteritems(), key=operator.itemgetter(1))[0]
        pos_dict[word] = most_likely_pos
    # Find the most common pos to assign to unknown words
    most_common_pos = max(pos_histogram.iteritems(), key=operator.itemgetter(1))[0]
    pos_dict['unk'] = most_common_pos
    return pos_dict

def wordshape(text):
    text_copy = copy.copy(text)
    t1 = re.sub('[A-Z]', 'X',text_copy)
    t2 = re.sub('[a-z]', 'x', t1)
    shape = re.sub('[0-9]', 'd', t2)
    if SIMPLE_WORDSHAPE:
        shape = wordshape_simplify(shape)
    if text=='<blank>':
        shape = '<blank>'
    return shape

def wordshape_simplify(text):
    if not text:
        return ""
    if len(text) == 1:
        return text
    if text[0] == text[1]:
        return wordshape_simplify(text[1:])
    return text[0] + wordshape_simplify(text[1:])

def write_results(data, fp):
    with open(fp, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerows(data)


if __name__ == '__main__':
    main()
