
import sys

file1 = sys.argv[1] # src in
file2 = sys.argv[2] # tgt in
file3 = sys.argv[3] # src out
file4 = sys.argv[4] # tgt out
unk = "<unk>"

f1 = open(file1, 'r')
f1_lines = f1.readlines()
f1.close()
f3 = open(file3, 'w')

f2 = open(file2, 'r')
f2_lines = f2.readlines()
f2.close()
f4 = open(file4, 'w')


if len(f1_lines) != len(f2_lines):
    sys.exit(f'Mismatch lines: {len(f1_lines)} vs {len(f2_lines)}')

for i in range(len(f1_lines)):
    line1 = f1_lines[i].strip()
    line2 = f2_lines[i].strip()
    
    if not line1 and not line2: # both empty
        continue
    elif not line1: # only src empty
        # print(line1 + ' <> ' + line2)
        print(unk, file=f3)
    elif not line2: # only tgt empty
        # print(line1 + ' <> ' + line2)
        print(unk, file=f4)
    else: # both not empty
        print(line1, file=f3)
        print(line2, file=f4)
