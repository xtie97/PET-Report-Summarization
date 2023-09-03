import os 
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CORENLP_HOME"] = "metrics/stanford-corenlp-full-2018-10-05"
os.system('pip install -U gin-config pyemd wmd blanc sentence-transformers stanza six sacrebleu')
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install -U transformers huggingface-hub tokenizers")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install requests==2.27.1")
# install java: apt update && apt install -y openjdk-11-jdk
os.system('apt update && apt install -y openjdk-11-jdk')
import nltk
nltk.download('stopwords')
nltk.download('punkt')

import pandas as pd
from tqdm import tqdm 
import numpy as np 
from metrics.meteor_metric import MeteorMetric
from metrics.bleu_metric import BleuMetric
from rouge_score import rouge_scorer
from metrics.chrfpp_metric import ChrfppMetric
from metrics.cider_metric import CiderMetric

def compute_metrics(df, save_path):
    #Import the pretrained model
    AI_impression = df['AI_impression'].tolist()
    impression = df['impressions'].tolist() 
    ### Rouge ###
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL', 'rougeLsum'], 
                                    use_stemmer=True, split_summaries=True)
    ### BLEU ###
    bleu = BleuMetric(lowercase=True)
    
    ### Meteor ### Need to install Java
    meteor = MeteorMetric() 
    
    ### CHRF ### 
    chrf_10 = ChrfppMetric(ncorder=10)
    ### Cider ###
    cider_4 = CiderMetric(n_gram=4)
   
    rouge1_f = []
    rouge2_f = []
    rouge3_f = []
    rougeL_f = []
    rougeLsum_f = []
    bleu_score = []
    meteor_score = []
    chrf_10_score = []
    cider_4_score = []
   
    for i in tqdm(np.arange(len(impression))):  
        gen_text = AI_impression[i]
        gt_text = impression[i]
        scores = rouge.score(gen_text, gt_text)
        #rouge1_p.append(list(scores['rouge1'])[0])
        #rouge1_r.append(list(scores['rouge1'])[1])
        rouge1_f.append(list(scores['rouge1'])[2]) 
        rouge2_f.append(list(scores['rouge2'])[2])
        rouge3_f.append(list(scores['rouge3'])[2])
        rougeL_f.append(list(scores['rougeL'])[2])
        rougeLsum_f.append(list(scores['rougeLsum'])[2])

    ### reference-dependent metrics ###
    bleu_dict = bleu.evaluate_batch(summaries=AI_impression, references=impression, aggregate=False)
    print('BLEU done')
    meteor_dict = meteor.evaluate_batch(summaries=AI_impression, references=impression, aggregate=False)
    print('Meteor done')
    chrf_10_dict = chrf_10.evaluate_batch(summaries=AI_impression, references=impression, aggregate=False)
    print('CHRF done')
    cider_4_dict = cider_4.evaluate_batch(summaries=AI_impression, references=impression, aggregate=False)
    print('Cider done')

    for i in tqdm(np.arange(len(impression))):  
        bleu_score.append(bleu_dict[i]['bleu'])
        meteor_score.append(meteor_dict[i]['meteor'])
        chrf_10_score.append(chrf_10_dict[i]['chrf'])
        cider_4_score.append(cider_4_dict[i]['cider'])

    df['rouge1_f'] = rouge1_f
    df['rouge2_f'] = rouge2_f
    df['rouge3_f'] = rouge3_f
    df['rougeL_f'] = rougeL_f
    df['rougeLsum_f'] = rougeLsum_f
    df['bleu_score'] = bleu_score
    df['meteor_score'] = meteor_score
    df['chrf_10_score'] = chrf_10_score
    df['cider_4_score'] = cider_4_score

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

        compute_metrics(df, save_path=f'test_cases/metrics/{filename}_metrics_lexion_overlap.xlsx') 
    

