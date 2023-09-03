import os 
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CORENLP_HOME"] = "metrics/stanford-corenlp-full-2018-10-05"
os.system('pip install -U gin-config pyemd wmd blanc sentence-transformers stanza six sacrebleu')
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install -U transformers huggingface-hub tokenizers")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install requests==2.27.1")
import nltk
nltk.download('stopwords')
nltk.download('punkt')

import pandas as pd
from tqdm import tqdm 
import numpy as np 
os.system('pip install scikit-learn==0.21.3') # for S3 metric (0.21.3, 0.19.2)
from metrics.s3_metric import S3Metric

def compute_metrics(df, save_path):
    #Import the pretrained model
    AI_impression = df['AI_impression'].tolist()
    impression = df['impressions'].tolist() 
    
    ### S3 metrics ### 
    s3 = S3Metric()
   
    s3_score_pyr = []
    s3_score_resp = []
   
    s3_dict = s3.evaluate_batch(summaries=AI_impression, references=impression, aggregate=False)
    print('S3 done')
  
    for i in tqdm(np.arange(len(impression))):  
        s3_score_pyr.append(s3_dict[i]['s3_pyr'])
        s3_score_resp.append(s3_dict[i]['s3_resp'])
       
    df['s3_score_pyr'] = s3_score_pyr
    df['s3_score_resp'] = s3_score_resp
 
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

        compute_metrics(df, save_path=f'test_cases/metrics/{filename}_metrics_supervised_regression.xlsx') 


