
import sys
import os
import logging
import argparse
import numpy as np
from fairseq.data import indexed_dataset

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('merge-binaries')

def merge_indexed_datasets(input_prefix, output_prefix, num_splits, bin_type, bin_dtype):
   
    path_to_bin = os.path.join(output_prefix + bin_type + '.bin')
    path_to_idx = os.path.join(output_prefix + bin_type + '.idx')
    ds = indexed_dataset.MMapIndexedDatasetBuilder(path_to_bin, dtype=bin_dtype)

    for i in range(1, num_splits + 1):
        logging.info('Processed subset: {} '.format(i))
        ds_idx_prefix = os.path.join(input_prefix + '.' + str(i) + bin_type)

        if indexed_dataset.dataset_exists(ds_idx_prefix, impl='mmap'):
            ds.merge_file_(ds_idx_prefix)
        else:
            sys.exit('No such indexed dataset {}'.format(ds_idx_prefix))
    ds.finalize(path_to_idx)

def merge_text_pos_mask(input_prefix, output_prefix, num_splits):
    ''' Merge the text, pos, mask files '''
    
    bin_pairs = [('', np.uint16), ('.pos', np.int16), ('.mask', np.float64)]
    for bin_type, bin_dtype in bin_pairs:
        merge_indexed_datasets(input_prefix, output_prefix, num_splits, bin_type, bin_dtype)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_prefix', type=str, required=True, help='Input file prefix name')
    parser.add_argument('--num_splits', type=int, required=True, help='Number of files to merge')
    parser.add_argument('--output_prefix', type=str, required=True, help='output file prefix')

    args = parser.parse_args()

    merge_text_pos_mask(args.input_prefix, args.output_prefix, args.num_splits)

if __name__ == '__main__':
    main()
