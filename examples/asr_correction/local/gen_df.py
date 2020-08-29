import pandas as pd
import csv, sys
test_cocnat = pd.read_csv(sys.argv[1])
model_out = pd.read_csv(sys.argv[2], sep='\t', names=['model_out_score', 'model_out'], header=None)
model_out['model_out_score'] = model_out['model_out_score'].astype(float)
model_out['model_out_score'] = model_out['model_out_score'] * -1
data_array = []
data_array.append(test_cocnat)
data_array.append(model_out)
df = pd.concat(data_array,axis=1)
df.to_csv('./test_results_with_ref.csv', index=False)
sys.path.insert(0, '/home/ec2-user/workspace/project/scripts/analysis/')
from calculate_wer import wer
def calculate_wer(row):
       row['one_best_wer'] = wer(row['gold_text'], row['one_best'])
       row['model_out_wer'] = wer(row['gold_text'], row['model_out'])
       row['model_wer_good_or_better'] = (row['model_out_wer'] <= row['one_best_wer'])
       return row
df = df.apply(calculate_wer, axis=1)
df.to_csv('./test_results_with_ref_with_wer.csv', index=False)
