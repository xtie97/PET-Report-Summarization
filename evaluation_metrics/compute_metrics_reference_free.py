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
from metrics.blanc_metric import BlancMetric
os.environ['PYTHONPATH'] = '/UserData/.../metrics'
from metrics.supert_metric import SupertMetric
from metrics.summa_qa_metric import SummaQAMetric 
from metrics.data_stats_metric import DataStatsMetric

def compute_metrics(df, save_path):
    #Import the pretrained model
    AI_impression = df['AI_impression'].tolist()
    impression = df['impressions'].tolist() 
    findings = df['findings'].tolist()
   
    ### Blanc ###
    blanc = BlancMetric(device='cuda', inference_batch_size=128, finetune_batch_size=32, use_tune=True)
    ### Supert metrics ###
    supert5 = SupertMetric(ref_metric='top5', sim_metric='f1')
    ### SummaQA ### length of findings are usually exceed 512 tokens 
    sQA = SummaQAMetric(batch_size=32, max_seq_len=512, use_gpu=True, tokenize=True)    
    ### Data Stats ###
    data_stats = DataStatsMetric(n_gram=3)

    blanc_score = []
    supert_score_top5 = []
    sQA_score = []
    length_score  = []
    density_score = []
    compression_score = []
    coverage_score = []
    novel_1_score = []
    novel_2_score = []
    novel_3_score = []
    repeat_1_score = []
    repeat_2_score = []
    repeat_3_score = []
  
    ### referece free metrics ###
    supert5_dict = supert5.evaluate_batch(summaries=AI_impression, input_texts=findings, aggregate=False)
    print('Supert done')
    #sQA_dict_ref = sQA.evaluate_batch(summaries=impression, input_texts=findings, aggregate=False)
    sQA_dict = sQA.evaluate_batch(summaries=AI_impression, input_texts=findings, aggregate=False)
    print('SummaQA done')
    #blanc_dict_ref = blanc.evaluate_batch(summaries=impression, input_texts=findings, aggregate=False)
    blanc_dict = blanc.evaluate_batch(summaries=AI_impression, input_texts=findings, aggregate=False) 
    print('Blanc done')
    
    data_stats_dict = data_stats.evaluate_batch(summaries=AI_impression, input_texts=findings, aggregate=False)
    
    for i in tqdm(np.arange(len(impression))):  
        length_score.append(data_stats_dict[i]['summary_length'])
        density_score.append(data_stats_dict[i]['density'])
        compression_score.append(data_stats_dict[i]['compression'])
        coverage_score.append(data_stats_dict[i]['coverage'])
        novel_1_score.append(data_stats_dict[i]['percentage_novel_1-gram'])
        novel_2_score.append(data_stats_dict[i]['percentage_novel_2-gram'])
        novel_3_score.append(data_stats_dict[i]['percentage_novel_3-gram'])
        repeat_1_score.append(data_stats_dict[i]['percentage_repeated_1-gram_in_summ'])
        repeat_2_score.append(data_stats_dict[i]['percentage_repeated_2-gram_in_summ'])
        repeat_3_score.append(data_stats_dict[i]['percentage_repeated_3-gram_in_summ'])
         
        supert_score_top5.append(supert5_dict[i]['supert'])
        sQA_score.append(sQA_dict[i]['summaqa_avg_fscore'])
        blanc_score.append(blanc_dict[i]['blanc'])
        
    df['length_score'] = length_score
    df['density_score'] = density_score
    df['compression_score'] = compression_score
    df['coverage_score'] = coverage_score
    df['novel_1_score'] = novel_1_score
    df['novel_2_score'] = novel_2_score
    df['novel_3_score'] = novel_3_score
    df['repeat_1_score'] = repeat_1_score
    df['repeat_2_score'] = repeat_2_score
    df['repeat_3_score'] = repeat_3_score

    # reference free metrics
    df['supert_score_top5'] = supert_score_top5
    df['sQA_score'] = sQA_score
    df['blanc_score'] = blanc_score  

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

        compute_metrics(df, save_path=f'test_cases/metrics/{filename}_metrics_ref_free.xlsx') 


