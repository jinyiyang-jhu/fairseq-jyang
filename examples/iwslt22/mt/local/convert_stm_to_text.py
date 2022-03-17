import sys
import os

stm = sys.argv[1]

def get_uttname(fname, idx, stime:float, etime:float):
    name = os.path.basename(fname).split(".")[0]
    stime = int(stime*1000)
    stime = f'{stime:07}'
    etime = int(etime*1000)
    etime = f'{etime:07}'
    return (idx + "_" + name + "_" + stime + "-" + etime)

if __name__ == "__main__":
    with open(stm, 'r') as sfh:
        for line in sfh:
            tokens = line.strip().split("\t")
            fname = tokens[0]
            uname = get_uttname(tokens[0], tokens[2], float(tokens[3]), float(tokens[4]))
            if len(tokens) < 7: # empty decoding result
                text = ""
            else:
                text = tokens[6]
            print (uname+' '+text)
            