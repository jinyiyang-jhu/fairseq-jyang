
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jieba
import sys

def main():
    ofile=sys.argv[1]
    with open (ofile, 'w') as output:
        for line in sys.stdin:
            print(" ".join(jieba.cut(line, cut_all=False)), file=output)

if __name__ == '__main__':
    main()

 