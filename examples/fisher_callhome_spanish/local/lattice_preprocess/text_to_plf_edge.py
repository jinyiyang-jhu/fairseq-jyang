# coding=utf-8
# Copyright (TBD)
# License (TBD)

from __future__ import print_function
import os
import argparse
import numpy as np

def convert_to_plf_edge(text, plf, add_bos=False, add_eos=False):
    with open(text, 'r') as ifile, open(plf, 'w') as ofile:
        for line in ifile:
            tokens = line.strip().split()
            uttid = tokens.pop(0)
            plf_edge = uttid + "\t" + "("
            if add_bos:
                plf_edge = plf_edge + "(('<s>', 0.0, 1),),"
            for word in tokens:
                edge = "(('{}', {}, 1),),".format(word, 0.0)
                plf_edge += edge
            if add_eos:
                plf_edge += "(('</s>', 0.0, 1),),"
            plf_edge += ")"
            print(plf_edge, file=ofile)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True, help='Input text file name')
    parser.add_argument('--plf', type=str, required=True, help='Output plf edge file name')
    parser.add_argument('--add_bos', action='store_true', help='Add <s> at the beginning of each output utterance')
    parser.add_argument('--add_eos', action='store_true', help='Add </s> at the end of each output utterance')
    args = parser.parse_args()

    convert_to_plf_edge(args.text, args.plf, add_bos=args.add_bos, add_eos=args.add_eos)


if __name__ == '__main__':
    main()
