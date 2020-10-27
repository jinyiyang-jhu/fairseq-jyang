# coding=utf-8
# Copyright (TBD)
# License (TBD)

"""
This script calculates the attention masks and positional encodings from lattices
for the transformer encoder.
"""
from __future__ import print_function
import argparse
import codecs
import numpy as np
from numpy import savez_compressed, asarray
from apply_bpe import read_vocabulary, BPE
from lattice_utils import LatticeReader, LatticeNode, Lattice
import sentencepiece as spm

def tokenization_lat(word_seqs, spm_processor, spm_format="piece"):
    """
    Tokenization for lattice text. Return the tokenized strings and id maps.
    Args:
    word_seqs (list): a list of words to be tokenized
    sp_processor: loaded sentencepiece processor
    spm_format (str): output tokenization format, "piece" or "id"
    """
    split_tokens = []
    split_map = []

    for i, word in enumerate(word_seqs):
        if spm_format == "piece":
            tokens = spm_processor.EncodeAsPieces(word)
        elif spm_format == "id":
            tokens = spm_processor.EncodeAsIds(word)
        split_tokens.extend([t for t in tokens])
        split_map.extend([(i, n) for n in range(len(tokens))])
    import pdb
    pdb.set_trace()
    return split_tokens, split_map

def load_dataset(spm_processor, lattice_file, lattice_bpe_file, spm_format="piece"):
#def load_dataset(spm_processor, lattice_file, lattice_bpe_file, lat_pos_file, lat_prob_mask_file,
 #                spm_format="piece", probabilistic_masks=True, mask_direction=None, linearize=False):
    """
    Compute the attention masks and positional encoding from lattice.
    """
    with open(lattice_file) as lat_file, open(lattice_bpe_file, 'w') as lat_bpe:
        output_pos = []
        output_mask = []
        lat_reader = LatticeReader()
        for i, line in enumerate(lat_file.readlines()):
            line = line.strip()
            tokens = line.split('\t')
            uttid = tokens.pop(0)
            lattice = lat_reader.read_sent(tokens[0], i)
            tokens, mapping = tokenization_lat(
                lattice.str_tokens(), spm_processor, spm_format=spm_format)
            nodes = []
            node_map = {}
            for j, (node_idx, n) in enumerate(mapping):
                if node_idx not in node_map:
                    node_map[node_idx] = []
                node_map[node_idx].append(j)

            for token, (node_idx, n) in zip(tokens, mapping):
                orig_node = lattice.nodes[node_idx]
                if n != 0:
                    nodes_prev = [len(nodes)-1]
                else:
                    nodes_prev = [node_map[node][-1]
                                  for node in orig_node.nodes_prev]
                if n != len(node_map[node_idx])-1:
                    nodes_next = [len(nodes)+1]
                else:
                    nodes_next = [node_map[node][0]
                                  for node in orig_node.nodes_next]
                node = LatticeNode(nodes_prev=nodes_prev, nodes_next=nodes_next, value=token,
                                   fwd_log_prob=orig_node.fwd_log_prob if n == 0 else 0,
                                   marginal_log_prob=orig_node.marginal_log_prob,
                                   bwd_log_prob=orig_node.bwd_log_prob if n == len(node_map[node_idx])-1 else 0)
                nodes.append(node)
            lattice_split = Lattice(idx=i, nodes=nodes)
            if linearize:
                pos = [i for i in range(len(lattice_split.nodes))]
            else:
                pos = lattice_split.longest_distances()
                pos = [p+1 for p in pos]
                log_conditional = lattice_split.compute_pairwise_log_conditionals(
                    mask_direction, probabilistic_masks)[0]
                print(uttid + '\t' + ' '.join(tokens), file=lat_bpe)
                output_pos.append(pos)
                output_mask.append(log_conditional)
        final_pos = {f'{i}': np.array(
            output_pos[i]) for i in range(len(output_pos))}
        final_mask = {f'{i}': np.array(
            output_mask[i]) for i in range(len(output_mask))}
        savez_compressed(lat_pos_file, **final_pos)
        savez_compressed(lat_prob_mask_file, **final_mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spm_format', type=str, default="piece",
                        help='Output spm format', choices=['piece', 'id'])
    parser.add_argument('--spm_model', type=str, required=True,
                        help='Trained sentencepiece model')
    parser.add_argument('--lat_ifile', type=str, required=True,
                        help='Input lattice PLF file name')
    parser.add_argument('--lat_bpe_file', type=str,
                        required=True, help='Output lattice bpe file name')
    #parser.add_argument('--lat_pos_file', type=str,
    #                    required=True, help='Output lattice postion indexes')
    #parser.add_argument('--lat_prob_mask', type=str,
    #                    required=True, help='Output lattice probabilistic mask')
    #parser.add_argument('--prob_mask_direction', type=str, default=None, choices=('None', 'fwd', 'bwd'),
    #                    help='Output lattice mask direction')

    args = parser.parse_args()
    #if args.prob_mask_direction == "None":
    #    mask_direction = None
    #else:
    #    mask_direction = args.prob_mask_direction
    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm_model)
    load_dataset(sp, args.lat_ifile, args.lat_bpe_file)
    #load_dataset(sp, args.lat_ifile, args.lat_bpe_file, args.lat_pos_file, args.lat_prob_mask,
    #             spm_format=args.spm_format, probabilistic_masks=True, mask_direction=mask_direction, linearize=False)


if __name__ == '__main__':
    main()
