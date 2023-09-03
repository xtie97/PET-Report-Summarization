import os 
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CORENLP_HOME"] = "metrics/stanford-corenlp-full-2018-10-05"
os.system('pip install -U gin-config pyemd wmd blanc sentence-transformers stanza six sacrebleu')
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install -U transformers huggingface-hub tokenizers")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install requests==2.27.1")
# install java: apt update && apt install -y openjdk-11-jdk
os.system('apt update && apt install -y openjdk-11-jdk')
os.system('pip install -U git+https://github.com/AIPHES/emnlp19-moverscore.git') # upgrade to the latest version 
import nltk
nltk.download('stopwords')
nltk.download('punkt')

import pandas as pd
from tqdm import tqdm 
import numpy as np 
from metrics.mover_score_metric import MoverScoreMetric
from metrics.bert_score_metric import BertScoreMetric
from metrics.rouge_we_metric import RougeWeMetric
os.environ['PYTHONPATH'] = '/UserData/.../metrics' # path to metrics folder


def compute_metrics(df, save_path):
    #Import the pretrained model
    AI_impression = df['AI_impression'].tolist()
    impression = df['impressions'].tolist() 
   
    ### Rouge We ###
    rouge_we1 = RougeWeMetric(n_gram=1)
    rouge_we2 = RougeWeMetric(n_gram=2)
    rouge_we3 = RougeWeMetric(n_gram=3)
    ### Mover Score ### 
    mover = MoverScoreMetric()
    ### Bert Score ###
    bert = BertScoreMetric(model_type='bert-large-uncased', num_layers=18)
  
    rougewe1_f = []
    rougewe2_f = []
    rougewe3_f = []
    bert_p = []
    bert_r = []
    bert_f = []
    mover_score = []
    meteor_score = []
  
    ### reference-dependent metrics ###
   
    bert_dict = bert.evaluate_batch(summaries=AI_impression, references=impression, aggregate=False)
    print('Bert done')
    mover_dict = mover.evaluate_batch(summaries=AI_impression, references=impression, aggregate=False)
    print('Mover done')
    rougewe1_dict = rouge_we1.evaluate_batch(summaries=AI_impression, references=impression, aggregate=False)
    rougewe2_dict = rouge_we2.evaluate_batch(summaries=AI_impression, references=impression, aggregate=False)
    rougewe3_dict = rouge_we3.evaluate_batch(summaries=AI_impression, references=impression, aggregate=False)
    print('Rouge-Wording Embedding done')

    for i in tqdm(np.arange(len(impression))):  
        bert_p.append(bert_dict[i]['bert_score_precision'])
        bert_r.append(bert_dict[i]['bert_score_recall'])
        bert_f.append(bert_dict[i]['bert_score_f1'])
        mover_score.append(mover_dict[i]['mover_score'])
        rougewe1_f.append(rougewe1_dict[i]['rouge_we_1_f'])
        rougewe2_f.append(rougewe2_dict[i]['rouge_we_2_f'])
        rougewe3_f.append(rougewe3_dict[i]['rouge_we_3_f'])
      
    df['rougewe1_f'] = rougewe1_f
    df['rougewe2_f'] = rougewe2_f
    df['rougewe3_f'] = rougewe3_f
    df['bert_f'] = bert_f
    df['mover_score'] = mover_score
    df['meteor_score'] = meteor_score
   
    # save to excel file
    df.to_excel(save_path, index=False)

if __name__ == '__main__':
    # Get data
    #text_file = ['PGN_200', 'Clinicallongformer2Roberta_200', 'BART_200', 'PEGASUS_200']
    text_file  = ['pgn-w-bg',
                  'clinicallongformer2roberta', 
                  'bart-large', 'biobart-large', 
                  'pegasus-large'
                  't5-large', 'clinical-t5-large', 'flan-t5-large', 'flan-t5-XL',
                  'gpt2-xl', 
                  'opt-1.3b'
                  'llama-7b-lora', 'alpaca-7b-lora'
                  ]
    
    os.makedirs('test_cases/metrics', exist_ok=True)
    for filename in text_file:
        # read excel file containing test cases along with model-generated impressions
        df = pd.read_excel(f'test_cases/{filename}_all.xlsx')
        # Clean text
        df = df[df['AI_impression'].notna()].reset_index(drop=True)
        df['impressions'] = df['impressions'].apply(lambda x: x.replace('\n',' '))
        df['AI_impression'] = df['AI_impression'].apply(lambda x: x.replace('\n',' '))
        df['findings'] = df['findings'].apply(lambda x: x.replace('\n',' '))
        df['findings_info'] = df['findings_info'].apply(lambda x: x.replace('\n',' '))

        compute_metrics(df, save_path=f'test_cases/metrics/{filename}_metrics_embedding.xlsx') 

