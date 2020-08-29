# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import Counter

from fairseq.tokenizer import tokenize_line
import torch
from fairseq.file_io import PathManager

class BinarizerMask:
    @staticmethod
    def binarize(
        npz,
        consumer,
        offset,
        end,
    ):
        nseq = 0
        for i in range(offset, end):
            mask = torch.DoubleTensor(npz[str(i)])
            consumer(mask)
            nseq += 1

        return {
            "nseq": nseq,
        }
        
    @staticmethod
    def find_offsets(loaded_filename, num_chunks):
        offsets = [0 for _ in range(num_chunks + 1)]
        len_loaded_file = len(loaded_filename)
        chunk_size = len_loaded_file // num_chunks
        for i in range(1, num_chunks):
            offsets[i] = chunk_size * i
        offsets[-1] = len_loaded_file
        return offsets