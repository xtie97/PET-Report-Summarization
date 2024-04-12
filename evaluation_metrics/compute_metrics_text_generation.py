import os 
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.system('pip install fairseq==0.9.0')
os.system('pip install -U gin-config pyemd wmd blanc sentence-transformers stanza six sacrebleu')
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import pandas as pd
from tqdm import tqdm 
import numpy as np 
from metrics.BARTScore.bart_score import BARTScorer
from metrics.prism.prism_sum import Prism

def compute_metrics(df, save_path):
    #Import the pretrained model
    AI_impression = df['AI_impression'].tolist()
    impression = df['impressions'].tolist() 
    findings = df['findings'].tolist()
    
    ### BART Score ###
    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large')
    bart_cnn_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
    bart_ParaBank_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn') # from GitHub
    bart_ParaBank_scorer.load(path='BARTScore/checkpoints/bart_score.pth')

    bart_PT_scorer = BARTScorer(device='cuda:0', checkpoint='BARTScore/checkpoints/bart-large')

    ### PEGASUS Score ###
    pegasus_PT_scorer = BARTScorer(device='cuda:0', checkpoint='BARTScore/checkpoints/pegasus-large')

    ### T5 Score ###
    t5_PT_scorer = BARTScorer(device='cuda:0', checkpoint='BARTScore/checkpoints/flan-t5')

    ### prism ###
    prism = Prism(model_dir='prism/checkpoints/m39v1', lang='en')
    print('Prism identifier:', prism.identifier()) # print the identifier of the model

    batch_size = 1
   
    bart_r = []
    bart_p = []
    bart_f = []
    bart_cnn_r = []
    bart_cnn_p = []
    bart_cnn_f = []
    bart_ParaBank_r = []
    bart_ParaBank_p = []
    bart_ParaBank_f = []
    bart_PT_r = []
    bart_PT_p = []
    bart_PT_f = []
    pegasus_PT_r = []
    pegasus_PT_p = []
    pegasus_PT_f = []
    t5_PT_r = []
    t5_PT_p = []
    t5_PT_f = []
    
    prism_score = []

    for i in tqdm(np.arange(len(impression))):  
        gen_text = AI_impression[i]
        gt_text = impression[i]
        gt_findings = findings[i]

        bart_p_ind = bart_scorer.score(srcs=[gt_text], tgts=[gen_text], batch_size=batch_size)[0] # precision
        bart_r_ind = bart_scorer.score(srcs=[gen_text], tgts=[gt_text], batch_size=batch_size)[0] # recall
        bart_f_ind = 2 * bart_p_ind * bart_r_ind / (bart_p_ind + bart_r_ind) # F1
        bart_r.append(bart_r_ind)
        bart_p.append(bart_p_ind)
        bart_f.append(bart_f_ind)

        #bart_cnn_faith_ind = bart_cnn_scorer.score(srcs=[gt_findings], tgts=[gen_text], batch_size=batch_size)[0] # faithfulness
        #bart_cnn_faith.append(bart_cnn_faith_ind)
        bart_cnn_p_ind = bart_cnn_scorer.score(srcs=[gt_text], tgts=[gen_text], batch_size=batch_size)[0] # precision
        bart_cnn_r_ind = bart_cnn_scorer.score(srcs=[gen_text], tgts=[gt_text], batch_size=batch_size)[0] # recall
        bart_cnn_f_ind = 2 * bart_cnn_p_ind * bart_cnn_r_ind / (bart_cnn_p_ind + bart_cnn_r_ind)  # F1
        
        bart_cnn_r.append(bart_cnn_r_ind)
        bart_cnn_p.append(bart_cnn_p_ind)
        bart_cnn_f.append(bart_cnn_f_ind)

        #bart_ParaBank_faith_ind = bart_ParaBank_scorer.score(srcs=[gt_findings], tgts=[gen_text], batch_size=batch_size)[0] # faithfulness
        #bart_ParaBank_faith.append(bart_ParaBank_faith_ind)
        bart_ParaBank_p_ind = bart_ParaBank_scorer.score(srcs=[gt_text], tgts=[gen_text], batch_size=batch_size)[0] # precision
        bart_ParaBank_r_ind = bart_ParaBank_scorer.score(srcs=[gen_text], tgts=[gt_text], batch_size=batch_size)[0] # recall
        bart_ParaBank_f_ind = 2 * bart_ParaBank_p_ind * bart_ParaBank_r_ind / (bart_ParaBank_p_ind + bart_ParaBank_r_ind)  # F1
        bart_ParaBank_r.append(bart_ParaBank_r_ind)
        bart_ParaBank_p.append(bart_ParaBank_p_ind)
        bart_ParaBank_f.append(bart_ParaBank_f_ind)

        bart_PT_p_ind = bart_PT_scorer.score(srcs=[gt_text], tgts=[gen_text], batch_size=batch_size)[0] # precision
        bart_PT_r_ind = bart_PT_scorer.score(srcs=[gen_text], tgts=[gt_text], batch_size=batch_size)[0] # recall
        bart_PT_f_ind = 2 * bart_PT_p_ind * bart_PT_r_ind / (bart_PT_p_ind + bart_PT_r_ind) # F1
        bart_PT_r.append(bart_PT_r_ind)
        bart_PT_p.append(bart_PT_p_ind)
        bart_PT_f.append(bart_PT_f_ind)

        pegasus_PT_p_ind = pegasus_PT_scorer.score(srcs=[gt_text], tgts=[gen_text], batch_size=batch_size)[0] # precision
        pegasus_PT_r_ind = pegasus_PT_scorer.score(srcs=[gen_text], tgts=[gt_text], batch_size=batch_size)[0] # recall
        pegasus_PT_f_ind = 2 * pegasus_PT_p_ind * pegasus_PT_r_ind / (pegasus_PT_p_ind + pegasus_PT_r_ind) # F1
        pegasus_PT_r.append(pegasus_PT_r_ind)
        pegasus_PT_p.append(pegasus_PT_p_ind)
        pegasus_PT_f.append(pegasus_PT_f_ind)

        t5_PT_p_ind = t5_PT_scorer.score(srcs=[gt_text], tgts=[gen_text], batch_size=batch_size)[0] # precision
        t5_PT_r_ind = t5_PT_scorer.score(srcs=[gen_text], tgts=[gt_text], batch_size=batch_size)[0] # recall
        t5_PT_f_ind = 2 * t5_PT_p_ind * t5_PT_r_ind / (t5_PT_p_ind + t5_PT_r_ind) # F1
        t5_PT_r.append(t5_PT_r_ind)
        t5_PT_p.append(t5_PT_p_ind)
        t5_PT_f.append(t5_PT_f_ind)
        
        prism_score.append( prism.score(cand=[gen_text], ref=[gt_text]) ) # segment_scores=True: Give score to each segments


    #df['bart_faith'] = bart_faith
    df['bart_p'] = bart_p
    df['bart_r'] = bart_r
    df['bart_f'] = bart_f
    #df['bart_cnn_faith'] = bart_cnn_faith
    df['bart_cnn_p'] = bart_cnn_p
    df['bart_cnn_r'] = bart_cnn_r
    df['bart_cnn_f'] = bart_cnn_f
    #df['bart_ParaBank_faith'] = bart_ParaBank_faith
    df['bart_ParaBank_p'] = bart_ParaBank_p
    df['bart_ParaBank_r'] = bart_ParaBank_r
    df['bart_ParaBank_f'] = bart_ParaBank_f
    df['bart_PT_p'] = bart_PT_p
    df['bart_PT_r'] = bart_PT_r
    df['bart_PT_f'] = bart_PT_f
    df['pegasus_PT_p'] = pegasus_PT_p
    df['pegasus_PT_r'] = pegasus_PT_r
    df['pegasus_PT_f'] = pegasus_PT_f
    df['t5_PT_p'] = t5_PT_p
    df['t5_PT_r'] = t5_PT_r
    df['t5_PT_f'] = t5_PT_f
    df['prism_score'] = prism_score
    
    # save to excel file
    df.to_excel(save_path, index=False)


if __name__ == '__main__':
    # Get data
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

        compute_metrics(df, save_path=f'test_cases/metrics/{filename}_metrics_text_generation.xlsx') 

