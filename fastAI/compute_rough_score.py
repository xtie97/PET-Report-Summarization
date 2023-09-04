# A simple example for getting ROUGE score, which is the most common metric for text summarization.
import os 
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install rouge-score==0.1.2") 
import pandas as pd
from tqdm import tqdm 
from rouge_score import rouge_scorer
import numpy as np 
import nltk
nltk.download('punkt')

def predict_text(df):
    AI_impression = df['AI_impression'].tolist()
    impression = df['impressions'].tolist() 
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL', 'rougeLsum'], use_stemmer=True, split_summaries=True)
    
    results_rouge1 = {'precision': [], 'recall': [], 'f': []} 
    results_rouge2 = {'precision': [], 'recall': [], 'f': []} 
    results_rouge3 = {'precision': [], 'recall': [], 'f': []} 
    results_rougeL = {'precision': [], 'recall': [], 'f': []} 
    results_rougeLsum = {'precision': [], 'recall': [], 'f': []} 

    for i in tqdm(np.arange(len(impression))):
        gen_text = AI_impression[i]
        gt_text = impression[i]
        
        # compute rouge score
        scores = scorer.score(gen_text, gt_text)
        results_rouge1['precision'].append(list(scores['rouge1'])[0]) 
        results_rouge1['recall'].append(list(scores['rouge1'])[1]) 
        results_rouge1['f'].append(list(scores['rouge1'])[2]) 

        results_rouge2['precision'].append(list(scores['rouge2'])[0]) 
        results_rouge2['recall'].append(list(scores['rouge2'])[1]) 
        results_rouge2['f'].append(list(scores['rouge2'])[2]) 

        results_rouge3['precision'].append(list(scores['rouge3'])[0]) 
        results_rouge3['recall'].append(list(scores['rouge3'])[1]) 
        results_rouge3['f'].append(list(scores['rouge3'])[2]) 

        results_rougeL['precision'].append(list(scores['rougeL'])[0]) 
        results_rougeL['recall'].append(list(scores['rougeL'])[1]) 
        results_rougeL['f'].append(list(scores['rougeL'])[2]) 

        results_rougeLsum['precision'].append(list(scores['rougeLsum'])[0])
        results_rougeLsum['recall'].append(list(scores['rougeLsum'])[1])
        results_rougeLsum['f'].append(list(scores['rougeLsum'])[2])
    
    results_rouge1['precision'] = np.around(np.mean(results_rouge1['precision']), 3)
    results_rouge1['recall'] = np.around(np.mean(results_rouge1['recall']), 3)
    results_rouge1['f'] = np.around(np.mean(results_rouge1['f']), 3)

    results_rouge2['precision'] = np.around(np.mean(results_rouge2['precision']), 3)
    results_rouge2['recall'] = np.around(np.mean(results_rouge2['recall']), 3)
    results_rouge2['f'] = np.around(np.mean(results_rouge2['f']), 3)

    results_rouge3['precision'] = np.around(np.mean(results_rouge3['precision']), 3)
    results_rouge3['recall'] = np.around(np.mean(results_rouge3['recall']), 3)
    results_rouge3['f'] = np.around(np.mean(results_rouge3['f']), 3)

    results_rougeL['precision'] = np.around(np.mean(results_rougeL['precision']), 3)
    results_rougeL['recall'] = np.around(np.mean(results_rougeL['recall']), 3)
    results_rougeL['f'] = np.around(np.mean(results_rougeL['f']), 3)

    results_rougeLsum['precision'] = np.around(np.mean(results_rougeLsum['precision']), 3)
    results_rougeLsum['recall'] = np.around(np.mean(results_rougeLsum['recall']), 3)
    results_rougeLsum['f'] = np.around(np.mean(results_rougeLsum['f']), 3)

    return results_rouge1['f'], results_rouge2['f'] , results_rouge3['f'], results_rougeL['f'], results_rougeLsum['f']


if __name__ == '__main__':
    # load prediction results 
    savename = 'pegasus_large'
    lr = 2e-4
    test_epoch = 10
    save_pred_dir = f'{savename}_pred'

    df = pd.read_excel(f'{save_pred_dir}/test_{lr}_{test_epoch}.xlsx')
    df['impressions'] = df['impressions'].apply(lambda x: x.replace('\n',' '))
    df['findings_info'] = df['findings_info'].apply(lambda x: x.replace('\n',' '))

    r1, r2, r3, rl, rlsum = predict_text(df)
      
