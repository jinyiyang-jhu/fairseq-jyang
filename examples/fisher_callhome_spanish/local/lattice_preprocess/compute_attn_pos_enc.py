# coding=utf-8
# Copyright (TBD)
# License (TBD)

"""
This script calculates the attention masks and positional encodings from lattices
for the transformer encoder.
"""
from __future__ import print_function
import sys
import os
import argparse
import codecs
from collections import Counter
import numpy as np
import logging
import torch
from apply_bpe import read_vocabulary, BPE
from lattice_utils import LatticeReader, LatticeNode, Lattice
from fairseq import tasks
from fairseq.data import indexed_dataset
from fairseq.tokenizer import tokenize_line

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('binarize-lattice')

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

def tokenization_lat(word_seq, bpe_tokenizer, apos_process=" &apos; "):
    """
    Tokenization for lattice text. Return the tokenized strings and id maps.
    Args:
    word_seq (list): a list of words (str)
    bpe_tokenizer: BPE object
    apos_process (bool): if True, convert apostrophe to &apos;
    """
    split_tokens = []
    split_map = []
    for i, word in enumerate(word_seq):
        if apos_process:
            word = word.replace("'", " &apos; ")
        tokens_str = bpe_tokenizer.process_line(line=word)
        tokens = tokens_str.split(' ')
        split_tokens.extend([t for t in tokens])
        split_map.extend([(i, n) for n in range(len(tokens))])
    return split_tokens, split_map

def compuate_and_binarize_dataset(lattice_file, dset_name, lat_utt_id, vocab, output_dir, bpe_tokenizer,
                probabilistic_masks=True, mask_direction=None, linearize=False, apos_process=" &apos; "):
    """
    Compute the attention masks and positional encoding from lattice.
    Binarize the dataset to Fairseq idx and bin files.
    """
    ds_text = indexed_dataset.make_builder(os.path.join(output_dir, dset_name + '.bin'),
        impl='mmap', vocab_size=len(vocab))
    ds_pos = indexed_dataset.MMapIndexedDatasetBuilder(os.path.join(output_dir, dset_name + '.pos.bin'),
        dtype=np.int16)
    ds_mask = indexed_dataset.MMapIndexedDatasetBuilder(os.path.join(output_dir, dset_name + '.mask.bin'),
        dtype=np.float64) 
    ntok = 0
    nunk = 0
    i = 0
    lat_reader = LatticeReader()
    with open(lattice_file) as lat_file, open(lat_utt_id, 'w') as f_uttid:
        for line in lat_file:
            line = line.strip()
            tokens = line.split('\t')
            uttid = tokens.pop(0)
            lattice = lat_reader.read_sent(tokens[0], i)
            tokens, mapping = tokenization_lat(lattice.str_tokens(), bpe_tokenizer, apos_process=apos_process)
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
            f_uttid.flush()
            ds_mask.add_item(torch.DoubleTensor(log_conditional))
            ds_pos.add_item(torch.IntTensor(pos))
            res = binarize_text(ds_text, ' '.join(tokens), vocab)
            logging.info("Processed utterance: {} , the number of BPE tokens is {} ".format(uttid, res['ntok']))
            ntok += res['ntok']
            nunk += res['nunk']
            i += 1
    ds_pos.finalize(os.path.join(output_dir, dset_name + '.pos.idx'))
    ds_mask.finalize(os.path.join(output_dir, dset_name + '.mask.idx'))
    ds_text.finalize(os.path.join(output_dir, dset_name + '.idx'))
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
    parser.add_argument('--apos_process', type=str, default=" &apos; ",help='Replace apostrophe with \" &apos; "')

    
    args = parser.parse_args()
    bpe_codes = codecs.open(args.bpe_code, encoding='utf-8')
    bpe_vocab = codecs.open(args.bpe_vocab, encoding='utf-8')
    bpe_vocab = read_vocabulary(bpe_vocab, args.bpe_vocab_thres)
    bpe_gloss = args.bpe_gloss

    lat_utt_id = os.path.join(args.output_dir, args.dset_name + ".lat.uttid")
    task = tasks.get_task('translation_lattice')
    vocab = task.load_dictionary(args.dict)
    
    logger.addHandler(logging.FileHandler(
        filename=os.path.join(args.output_dir, 'binarize_lattice.log'),
    ))
    logger.info(args)

    if args.prob_mask_direction == "None":
        mask_direction = None
    else:
        mask_direction = args.prob_mask_direction

    with open(bpe_gloss, 'r') as f:
        glossaries = f.readlines()
    glossaries = [x.strip() for x in glossaries]
    bpe_tokenizer = BPE(codes=bpe_codes, vocab=bpe_vocab, glossaries=glossaries)

    compuate_and_binarize_dataset(args.lat_ifile, args.dset_name, lat_utt_id, vocab, args.output_dir, bpe_tokenizer,
        probabilistic_masks=True, mask_direction=mask_direction, linearize=False, apos_process=args.apos_process)

if __name__ == '__main__':
    main()
