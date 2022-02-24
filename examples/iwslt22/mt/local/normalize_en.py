
import io
import re
import sys

english_filter = re.compile(r'\(|\)|\#|\+|\=|\?|\!|\;|\.|\,|\"|\:')

def normalize_text(utterance):
    return re.subn(english_filter, '', utterance)[0].lower()

def main():
    infile = sys.argv[1]
    with  open(infile, encoding='utf8') as f:
        for line in f:
            text = normalize_text(line.strip())
            print(text, file=sys.stdout)

if '__name__' == '__main__':
    main()