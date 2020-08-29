import multiprocessing as mp
import sys
 
import numpy as np
 
def func(val, basename):
   nf = np.load(val)
   print ("\nFile length is {}\n".format(len(nf)))
   with open(basename, 'w') as f:
     print(len(nf), file=f)
   return len(nf)
 
if __name__ == '__main__':
   inputfile = sys.argv[1]
   basename = 'tmp'
   cpu_count = mp.cpu_count()
   pool = mp.Pool(processes = cpu_count)
 
   results = []
   num = 1
   while cpu_count >= 1:
       results.append(pool.apply_async(func, (inputfile,basename+str(num),)))
       cpu_count = cpu_count - 1
       num = num + 1
 
   output = [p.get() for p in results]
   print (output)
 
   pool.close()
   pool.join()
