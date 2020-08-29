 
import sys
from multiprocessing import Pool
import numpy as np
 
def get_offsets(loaded_filename, num_chunks):
   loaded_filename = np.load(loaded_filename)
   offsets = [0 for _ in range(num_chunks + 1)]
   len_loaded_file = len(loaded_filename)
   chunk_size = len_loaded_file // num_chunks
   for i in range(1, num_chunks):
       offsets[i] = chunk_size * i
   offsets[-1] = len_loaded_file
   return offsets
 
#def count_size(npz, output_file):
def count_size():
   n = 0
   print('Processing ')
   #npz = np.load(npz)
   #with open(output_file, 'w') as f:
   #  for i in range(len(npz)):
   #      np_shape = str(npz[str(i)].size)
   #      print('size is {}'.format(np_shape), file=f)
   #      n += 1
   return n
 
def main():
 npz = sys.argv[1]
 output_file = sys.argv[2]
 num_workers = int(sys.argv[3])
 
 pool = None
 pool = Pool(processes=num_workers)
 #offsets = get_offsets(npz, num_workers)
 total_counts = [0]
 
 def merge(worker_result):
   total_counts[0] += worker_result
 
 for worker_id in range(1, num_workers+1):
   tmp_file = output_file + str(worker_id)
   pool.apply_async(
     count_size,
     (
       str(worker_id) + '.' + npz,
       tmp_file,
     ),
     callback=merge,
   )
 pool.close()
 pool.join()
 
if __name__ == "__main__":
   main()
