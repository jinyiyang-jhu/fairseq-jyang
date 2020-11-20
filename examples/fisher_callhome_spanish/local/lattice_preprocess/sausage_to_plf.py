# coding=utf-8
# Copyright (TBD)
# License (TBD)

from __future__ import print_function
import os
import argparse
import numpy as np

def read_word_sym_table(word_sym_table):
    dict_int2word = {}
    with open(word_sym_table, 'r') as word2int:
        for line in word2int:
            tokens = line.strip().split()
            dict_int2word[tokens[1]] = tokens[0]
    return dict_int2word

def convert_to_plf_edge_remove_eps(ifile, ofile, int2word, eps_id="0", add_bos=True, add_eos=True):
    with open(ifile, 'r') as sau, open(ofile, 'w') as plf:
        for line in sau: # line is each utterance
            line = line.strip().replace(']', '')
            tokens = line.split('[')
            uttid = tokens.pop(0) + " " # two spaces 
            plf_edge = ""
            empty_flag = True
            for edges in tokens: # edges is each time bin
                edges_tokens = edges.split()
                edge = ""
                if len(edges_tokens) == 2 and edges_tokens[0] == eps_id:
                    continue
                empty_flag = False
                for idx, item in enumerate(edges_tokens):
                    if idx % 2 ==  0:
                        if idx == 0:
                            edge = "(" + edge
                        edge += "('{}', {}, 1),".format(int2word[item], np.log(float(edges_tokens[idx+1])))
                    elif idx == len(edges_tokens) - 1:
                        edge += "),"
                plf_edge += edge
            if add_bos:
                plf_edge = "((('<s>', 0.0, 1),)," + plf_edge
            else:
                plf_edge = "(" + plf_edge
            if add_eos:
                plf_edge += "(('</s>', 0.0, 1),),"
            plf_edge += ")"
            if empty_flag:
                print("Empty line for {}".format(uttid))
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

    args = parser.parse_args()

    int2word = read_word_sym_table(args.word_sym_table)
    convert_to_plf_edge_remove_eps(args.sausage, args.plf, int2word, 
        eps_id=args.eps_id, add_bos=args.add_bos, add_eos=args.add_eos)


if __name__ == '__main__':
    main()
