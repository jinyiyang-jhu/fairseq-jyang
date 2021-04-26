
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This script preprocess the raw text in three steps:
1) convert letters to lower case
2) (for Chinese) text segmentation
"""

import jieba
import sys
import argparse

def main(args):
    ofile=args.output
    with open (ofile, 'w') as output:
        for line in sys.stdin:
            line = line.strip().lower()
            if args.lan == "zh":
                print(" ".join(jieba.cut(line, cut_all=False)), file=output)
            else:
                print(line, file=output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Take in raw text, apply all preprocessing.')
    parser.add_argument('--lan', help='Language')
    parser.add_argument('--output', help='Output filename')
    args = parser.parse_args()
    main(args)

 