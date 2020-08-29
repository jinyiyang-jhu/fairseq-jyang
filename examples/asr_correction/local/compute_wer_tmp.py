import sys
import numpy
 
def wer(gold, hypothesis):
   """
   Calculation of WER with Levenshtein distance.
 
   Works only for iterables up to 254 elements (uint8).
   O(nm) time ans space complexity.
 
   Parameters
   ----------
   r : string
   h : string
 
   Returns
   -------
   int -> wer(r, h)/len(r)
 
   Examples
   --------
   >>> wer("who is there".split(), "is there".split())
   1
   >>> wer("who is there".split(), "".split())
   3
   >>> wer("".split(), "who is there".split())
   3
   """
 
   try:
       r = re_format(hypothesis, gold).split()
   except Exception as e:
       raise TypeError('error in gold text: {}'.format(gold_text))
 
   try:
       h = hypothesis.split()
   except Exception as e:
       raise TypeError('error in hypothesis text: {}'.format(hypothesis))
 
   d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
   d = d.reshape((len(r)+1, len(h)+1))
   for i in range(len(r)+1):
       for j in range(len(h)+1):
           if i == 0:
               d[0][j] = j
           elif j == 0:
               d[i][0] = i
 
   # computation
   for i in range(1, len(r)+1):
       for j in range(1, len(h)+1):
           if r[i-1] == h[j-1]:
               d[i][j] = d[i-1][j-1]
           else:
               substitution = d[i-1][j-1] + 1
               insertion    = d[i][j-1] + 1
               deletion     = d[i-1][j] + 1
               d[i][j] = min(substitution, insertion, deletion)
 
   return (d[len(r)][len(h)])/len(r)
 
 
def calculate_wer():
   p_file = sys.argv[1]
   g_file = sys.argv[2]
 
   fp = open(p_file)
   fg = open(g_file)
 
   plines = fp.readlines()
   glines = fg.readlines()
 
   total_error = 0
   total_lines = len(plines)
 
   for i, pl in enumerate(plines):
       total_error += wer(glines[i], pl)
 
   error = total_error / total_lines
   print("wer for {} is {}".format(p_file, error))
 
 
def re_format(asr_out, old_gold):
 
   gold_text = old_gold
 
   try:
       if ("what's" in asr_out and "what is" in gold_text):
           gold_text = gold_text.replace("what is", "what's")
       elif ("what is" in asr_out and "what's" in gold_text):
           gold_text = gold_text.replace("what's", "what is")
 
       if ("how's" in asr_out and "how is" in gold_text):
           gold_text = gold_text.replace("how is", "how's")
       elif ("how is" in asr_out and "how's" in gold_text):
           gold_text = gold_text.replace("how's", "how is")
 
       if ("where's" in asr_out and "where is" in gold_text):
           gold_text = gold_text.replace("where is", "where's")
       elif ("where is" in asr_out and "where's" in gold_text):
           gold_text = gold_text.replace("where's", "where is")
 
       if ("when's" in asr_out and "when is" in gold_text):
           gold_text = gold_text.replace("when is", "when's")
       elif ("when is" in asr_out and "when's" in gold_text):
           gold_text = gold_text.replace("when's", "when is")
 
       if ("who's" in asr_out and "who is" in gold_text):
           gold_text = gold_text.replace("who is", "who's")
       elif ("who is" in asr_out and "who's" in gold_text):
           gold_text = gold_text.replace("who's", "who is")
 
       if ("i'm" in asr_out and "i am" in gold_text):
           gold_text = gold_text.replace("i am", "i'm")
       elif ("i am" in asr_out and "i'm" in gold_text):
           gold_text = gold_text.replace("i'm", "i am")
 
       if ("he's" in asr_out and "he is" in gold_text):
           gold_text = gold_text.replace("he is", "he's")
       elif ("he is" in asr_out and "he's" in gold_text):
           gold_text = gold_text.replace("he's", "he is")
 
       if ("you're" in asr_out and "you are" in gold_text):
           gold_text = gold_text.replace("you are", "you're")
       elif ("you are" in asr_out and "you're" in gold_text):
           gold_text = gold_text.replace("you're", "you are")
 
       if ("she's" in asr_out and "she is" in gold_text):
           gold_text = gold_text.replace("she is", "she's")
       elif ("she is" in asr_out and "she's" in gold_text):
           gold_text = gold_text.replace("she's", "she is")
 
       if ("it's" in asr_out and "it is" in gold_text):
           gold_text = gold_text.replace("it is", "it's")
       elif ("it is" in asr_out and "it's" in gold_text):
           gold_text = gold_text.replace("it's", "it is")
 
   except Exception as e:
       print(gold_text)
       print(e)
       raise TypeError('some error in gold text: {}'.format(gold_text))
 
   return gold_text
 
if __name__ == '__main__':
   calculate_wer()
 