 
# This script checks the last column of prob mask, to see whether the exponential of them are close to 1.
import argparse
 
import tqdm
import numpy as np
 
def main():
 parser = argparse.ArgumentParser()
 parser.add_argument('mask_npz', type=str, help='input file (npz) for positional indices')
 args = parser.parse_args()
 mask = np.load(args.mask_npz)
 mask_check = np.zeros((len(mask), 4)) # max_of_first_col, min_of_first_col, max_of_last_col, min_of_last_col
 
 for i in tqdm.tqdm(mask.keys()):
   last_col = np.exp(mask[i][:,-1])
   first_col = np.exp(mask[i][:, 0])
   i = int(i)
   mask_check[i][0] = max(first_col)
   mask_check[i][1] = min(first_col)
   mask_check[i][2] = max(last_col)
   mask_check[i][3] = min(last_col)
 
 print('Total max: ' + str(np.argmax(mask_check)) + ' : ' + str(np.max(mask_check)))
 print('Total min: ' + str(np.argmin(mask_check)) + ' : ' + str(np.min(mask_check)))
 
if __name__ == "__main__":
 main()

