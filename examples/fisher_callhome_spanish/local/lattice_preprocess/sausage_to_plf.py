# coding=utf-8
# Copyright (TBD)
# License (TBD)

from __future__ import print_function
import os
import argparse
import numpy as np
import sys

def read_word_sym_table(word_sym_table):
    dict_int2word = {}
    with open(word_sym_table, 'r') as word2int:
        for line in word2int:
            tokens = line.strip().split()
            tokens[0] = tokens[0].replace("'", "\\'")
            dict_int2word[tokens[1]] = tokens[0]
    return dict_int2word

def check_skip_token(token, skip_token=None):
    if token == skip_token:
        token = "<pad>"
    return token

def prune_sausage(current_bin, threshold=1):
    '''Args:
    This function prunes the sausage, keeping the fewest candidates that sum up above threshold.
    current_bin (list): format: [word-id-1 score-1 word-id-2 score-2 ... ]
    threshold (float): a non-negative number between 0 and 1
    '''
    if threshold < 1 and threshold >= 0:
        words = []
        scores = []
        for idx, item in enumerate(current_bin):
            if idx %2 == 0:
                words.append(item)
            else:
                scores.append(float(item))
        scores = np.array(scores)
        sorted_scores = np.sort(scores)[::-1]
        sorted_indices = np.argsort(scores)[::-1]
        selected_indices = np.where(sorted_scores.cumsum() >= threshold)[0][0]+1
        new_arr = []
        for idx in sorted_indices[:selected_indices]:
            new_arr.append(words[idx])
            new_arr.append(scores[idx])
        return new_arr
    elif threshold == 1:
        return current_bin
    else:
        sys.exit('Invalid value of prune threshold: {}'.format(threshold))

def convert_to_plf_edge_remove_eps(ifile, ofile, int2word, eps_id="0", 
    add_bos=True, add_eos=True, eps=1e-20, skip_token=None, threshold=1):
    with open(ifile, 'r') as sau, open(ofile, 'w') as plf:
        for line in sau: # line is each utterance
            line = line.strip().replace(']', '')
            tokens = line.split('[')
            uttid = tokens.pop(0) + "\t" + "(" 
            plf_edge = ""
            empty_flag = True
            for edges in tokens: # edges is each time bin
                edges_tokens = edges.split()
                edge = ""
                edges_tokens = prune_sausage(edges_tokens, threshold)
                if len(edges_tokens) == 2 and edges_tokens[0] == eps_id: # only <eps> in this bin
                    continue
                empty_flag = False
                for idx, item in enumerate(edges_tokens):
                    if idx % 2 ==  0: # word-id
                        if idx == 0:
                            edge = "(" + edge
                        edge += "('{}', {}, 1),".format(check_skip_token(int2word[item], skip_token=skip_token), 
                                                    np.log(eps + float(edges_tokens[idx+1])))
                    elif idx == len(edges_tokens) - 1:
                        edge += "),"
                plf_edge += edge
            if add_bos:
                plf_edge = "(('<s>', 0.0, 1),)," + plf_edge
            if add_eos:
                plf_edge += "(('</s>', 0.0, 1),),"
            plf_edge += ")"
            if empty_flag:
                print(uttid + "(('<s>', 0.0, 1),),(('<unk>', 0.0, 1),),(('</s>', 0.0, 1),),)", file=plf)
                print(uttid + plf_edge, file=plf)
                print("empty line for {}".format(uttid))
            else:
                print(uttid + plf_edge, file=plf)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sausage', type=str, required=True, help='Input sausage file name')
    parser.add_argument('--plf', type=str, required=True, help='Output plf edge file name')
    parser.add_argument('--word_sym_table', type=str, required=True, 
        help='A file containing the mapping from word id to word symbol')
    parser.add_argument('--remove_eps', action='store_true', help='Remove links with only <eps>')
    parser.add_argument('--eps_id', type=str, default="0", help='Word id for <eps>')
    parser.add_argument('--add_bos', action='store_true', help='Add <s> at the beginning of each output utterance')
    parser.add_argument('--add_eos', action='store_true', help='Add </s> at the end of each output utterance')
    parser.add_argument('--skip_token', default=None, help='Skip the token for model training; it will be replaced by "<pad>"')
    parser.add_argument('--eps', default=1e-20, help='Epsilon to avoid np log zero division error')
    parser.add_argument('--prune_threshold', type=float, help='Keep the edges that sum up above the threshold')
    args = parser.parse_args()

    assert args.prune_threshold <= 1 and args.prune_threshold >= 0
    int2word = read_word_sym_table(args.word_sym_table)
    convert_to_plf_edge_remove_eps(args.sausage, args.plf, int2word, 
        eps_id=args.eps_id, add_bos=args.add_bos, add_eos=args.add_eos, eps=args.eps, 
        skip_token=args.skip_token, threshold=args.prune_threshold)


if __name__ == '__main__':
    main()
