import os 
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install bert-score fastai==2.7.11") 
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install scipy uvicorn gunicorn==19.9.0 fastapi uvloop httptools")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install ohmeow-blurr==1.0.5")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install -U huggingface-hub transformers tokenizers")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install requests==2.27.1")
import nltk
nltk.download('punkt')

from metrics.UniEval.utils import convert_to_json
from metrics.UniEval.evaluator import get_evaluator
import pandas as pd


def compute_metrics(df, save_path):
    task = 'summarization'
    coherence = []
    consistency = []
    fluency = []
    relevance = []
    overall = []
        
    # a list of source documents
    src_list = df['findings'].tolist() 
    # a list of human-annotated reference summaries
    ref_list = df['impressions'].tolist() 
    # a list of model outputs to be evaluataed
    output_list = df['AI_impression'].tolist() 

    # Prepare data for pre-trained evaluators
    data = convert_to_json(output_list=output_list, src_list=src_list, ref_list=ref_list)
    # Initialize evaluator for a specific task
    evaluator = get_evaluator(task)
    # Get multi-dimensional evaluation scores
    eval_scores = evaluator.evaluate(data, dims=['coherence', 'consistency', 'fluency', 'relevance'], 
                                    overall=True, print_result=True)
    for kk in range(len(eval_scores)):
        coherence.append(eval_scores[kk]['coherence'])
        consistency.append(eval_scores[kk]['consistency'])
        fluency.append(eval_scores[kk]['fluency'])
        relevance.append(eval_scores[kk]['relevance'])
        overall.append(eval_scores[kk]['overall']) 

    df['coherence'] = coherence
    df['consistency'] = consistency
    df['fluency'] = fluency
    df['relevance'] = relevance
    df['overall'] = overall
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

        compute_metrics(df, save_path=f'test_cases/metrics/{filename}_metrics_unieval.xlsx') 
   

