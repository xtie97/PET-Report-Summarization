import os
os.system('pip install radgraph f1chexbert')
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install -U transformers huggingface-hub tokenizers")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install requests==2.27.1")
os.system("cp metrics/radgraph/chexbert.pth /root/.cache/chexbert/")
os.system("cp metrics/radgraph/radgraph.tar.gz /root/.cache/chexbert/")
from radgraph import F1RadGraph
import pandas as pd
import nltk
nltk.download('punkt')
 
def compute_metrics(df, save_path):
    #f1chexbert = F1CheXbert(device="cuda")
    #Import the pretrained model
    refs = df['impressions'].tolist() 
    # a list of model outputs to be evaluataed
    hyps = df['AI_impression'].tolist() 
    f1radgraph = F1RadGraph(reward_level="partial")
    f1radgraph_score = [] 
    for hyp, ref in zip(hyps, refs):
        # check if hyp is a list
        if isinstance(hyp, str):
            hyp = [hyp]
        # check if ref is a list
        if isinstance(ref, str):
            ref = [ref]
        score, _, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=hyp,
                                                                               refs=ref)
        f1radgraph_score.append(score)
        
    df['RadGraph'] = f1radgraph_score
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

        compute_metrics(df, save_path=f'test_cases/metrics/{filename}_metrics_radgraph.xlsx') 
   