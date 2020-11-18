# coding=utf-8
# Copyright (TBD)
# License (TBD)

from __future__ import print_function
import os
import argparse

def convert_to_plf_edge_remove_eps(ifile, ofile, int2word, eps_id=0):
    with open(ifile, 'r') as sau, open(ofile, 'w') as plf:
        for line in sau: # line is each utterance
            line = line.strip().replace(']', '')
            tokens = line.split('[')
            uttid = tokens.pop(0) # with one space at the end
            for node_id, item in enumerate(tokens): # item is each time bin
                for idx, bin in enumerate(item):
                    

                    









def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sausage_ifile', type=str, required=True, help='Input sausage file name')
    parser.add_argument('--plf_ofile', type=str, required=True, help='Output plf edge file name')
    parser.add_argument('--int2word', type=str, required=True, 
        help='A file containing the mapping from word id to word symbol')
    parser.add_argument('--remove-eps', action='store_true', help='Remove links with only <eps>')
    parser.add_argument('--eps_id', type=int, default=0, help='Word id for <eps>')
    parser.add_argument('--add_eos_bos', action='store_true', help='Add <s> and </s> to each output utterance')

    args = parser.parse_args()
    convert_to_plf_edge(args.sausage_ifile, args.plf_ofile, args.int2word, eps_id=args.eps_id)


if __name__ == '__main__':
    main()
