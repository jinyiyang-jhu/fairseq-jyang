# coding=utf-8
# Copyright (TBD)
# License (TBD)

"""
This script calculates the attention masks and positional encodings from lattices
for the transformer encoder.
"""
from __future__ import print_function
import os
import argparse
import codecs
from collections import Counter
import numpy as np
import torch
from apply_bpe import read_vocabulary, BPE
from lattice_utils import LatticeReader, LatticeNode, Lattice
from fairseq import tasks
from fairseq.data import indexed_dataset
from fairseq.tokenizer import tokenize_line

def binarize_text(ds, words, dict):
    replaced = Counter()
    ntok = 0
    def replaced_consumer(word, idx):
        if idx == dict.unk_index and word != dict.unk_word:
            replaced.update([word])
    ids = dict.encode_line(
        line=words,
        line_tokenizer=tokenize_line,
        add_if_not_exist=False,
        consumer=replaced_consumer,
        append_eos=False,
        reverse_order=False,
    )
    ds.add_item(ids)
    ntok += len(ids)
    return {
        "nunk": sum(replaced.values()),
        "ntok": ntok
    }

def tokenization_lat(word_seq, bpe_codes, bpe_vocab, bpe_gloss):
    """
    Tokenization for lattice text. Return the tokenized strings and id maps.
    Args:
    word_seq (list): a list of words (str)
    bpe_codes (codecs.StreamReaderWriter): bpe code file
    bpe_vocab (codecs.StreamReaderWriter): bpe vocabulary file
    bpe_gloss (str): a file contains bpe glossaries
    """
    split_tokens = []
    split_map = []

    with open(bpe_gloss, 'r') as f:
        glossaries = f.readlines()
    glossaries = [x.strip() for x in glossaries]
    bpe = BPE(codes=bpe_codes, vocab=bpe_vocab, glossaries=glossaries)

    for i, word in enumerate(word_seq):
        tokens_str = bpe.process_line(line=word)
        tokens = tokens_str.split(' ')
        split_tokens.extend([t for t in tokens])
        split_map.extend([(i, n) for n in range(len(tokens))])
    return split_tokens, split_map

def compuate_and_binarize_dataset(lattice_file, dset_name, lat_utt_id, vocab, txt_dir, pos_dir, mask_dir,
                bpe_codes, bpe_vocab, bpe_gloss, probabilistic_masks=True, 
                mask_direction=None, linearize=False):
    """
    Compute the attention masks and positional encoding from lattice.
    Binarize the dataset to Fairseq idx and bin files.
    """
    ds_text = indexed_dataset.make_builder(os.path.join(txt_dir, dset_name + '.bin'),
        impl='mmap', vocab_size=len(vocab))
    ds_pos = indexed_dataset.MMapIndexedDatasetBuilder(os.path.join(pos_dir, dset_name + '.pos.bin'),
        dtype=np.int16)
    ds_mask = indexed_dataset.MMapIndexedDatasetBuilder(os.path.join(pos_dir, dset_name + '.mask.bin'),
        dtype=np.float64) 
    ntok = 0
    nunk = 0
    with open(lattice_file) as lat_file, open(lat_utt_id, 'w') as f_uttid:
        lat_reader = LatticeReader()
        for i, line in enumerate(lat_file.readlines()):
            line = line.strip()
            tokens = line.split('\t')
            uttid = tokens.pop(0)
            lattice = lat_reader.read_sent(tokens[0], i)
            tokens, mapping = tokenization_lat(
                lattice.str_tokens(), bpe_codes, bpe_vocab, bpe_gloss)
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
            log_conditional = lattice_split.compute_pairwise_log_conditionals(mask_direction, probabilistic_masks)[0]
            print(uttid + ' ' + str(i), file=f_uttid)
            ds_mask.add_item(torch.DoubleTensor(log_conditional))
            ds_pos.add_item(torch.IntTensor(pos))
            res = binarize_text(ds_text, ' '.join(tokens), vocab)
            ntok += res['ntok']
            nunk += res['nunk']
    ds_pos.finalize(os.path.join(pos_dir, dset_name + '.pos.idx'))
    ds_mask.finalize(os.path.join(mask_dir, dset_name + '.mask.idx'))
    ds_text.finalize(os.path.join(txt_dir, dset_name + '.idx'))
    print('Number of unknown tokens is {} / {}'.format(nunk, ntok))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lat_ifile', type=str, required=True, help='Input lattice PLF file name')
    parser.add_argument('--dset_name', type=str, required=True, help='Name of the input dataset')
    parser.add_argument('--output_dir', type=str, required=True, 
        help='Output directory for encoded PLF, positional indice and probability matrice')
    parser.add_argument('--prob_mask_direction', type=str, default=None, choices=('None', 'fwd', 'bwd'),
                        help='Output lattice mask direction')
    parser.add_argument('--dict', type=str, required=True, help='Dictionary used for convert words to word-ids')
    parser.add_argument('--bpe_code', type=str, required=True, help='Code file for BPE')
    parser.add_argument('--bpe_vocab', type=str, required=True,  help='BPE vocabulary')
    parser.add_argument('--bpe_gloss', type=str, required=True, help='BPE glossaries terms')
    parser.add_argument('--bpe_vocab_thres', type=str, default=1,help='BPE vocabulary frequency threshod')
    
    args = parser.parse_args()
    bpe_codes = codecs.open(args.bpe_code, encoding='utf-8')
    bpe_vocab = codecs.open(args.bpe_vocab, encoding='utf-8')
    bpe_vocab = read_vocabulary(bpe_vocab, args.bpe_vocab_thres)
    bpe_gloss = args.bpe_gloss
    output_dir= args.output_dir

    txt_dir = os.path.join(output_dir, "text")
    pos_dir = os.path.join(output_dir, "positions")
    mask_dir = os.path.join(output_dir, "masks")
    lat_utt_id = os.path.join(output_dir, "uttids.txt")
    task = tasks.get_task('translation_lattice')
    vocab = task.load_dictionary(args.dict)

    for d in [txt_dir, pos_dir, mask_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    if args.prob_mask_direction == "None":
        mask_direction = None
    else:
        mask_direction = args.prob_mask_direction

    compuate_and_binarize_dataset(args.lat_ifile, args.dset_name, lat_utt_id, vocab, txt_dir, pos_dir, mask_dir,
        bpe_codes, bpe_vocab, bpe_gloss, probabilistic_masks=True,
        mask_direction=mask_direction, linearize=False)

if __name__ == '__main__':
    main()
