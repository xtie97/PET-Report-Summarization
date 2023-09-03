import os 
os.environ['CURL_CA_BUNDLE'] = ''
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install bert-score fastai==2.7.11") 
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install scipy uvicorn gunicorn==19.9.0 fastapi uvloop httptools")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install ohmeow-blurr==1.0.5")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install huggingface-hub --upgrade")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install -U transformers")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install requests==2.27.1")

import pandas as pd
from tqdm import tqdm 
import torch
from fastai.text.all import *
from transformers import *
from blurr.text.data.all import *
from blurr.text.modeling.all import *
import datasets
import time 

def predict_text(reports, model_template, savename, test_epoch):
    #Import the pretrained model
    tokenizer = AutoTokenizer.from_pretrained(model_template.split('/')[:-1]) # root folder save the tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_template).eval().to('cuda')
    #Create mini-batch and define parameters
    encoder_max_length = 1024
    decoder_max_length = 512
   
    # map data correctly
    def generate_summary(batch):
        # Tokenizer will automatically set [BOS] <text> [EOS]
        inputs = tokenizer(batch["findings_info"], padding="max_length", truncation=True, max_length=encoder_max_length, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=decoder_max_length, 
                                                                           num_beam_groups=1,
                                                                           num_beams=4, 
                                                                           do_sample=False,
                                                                           diversity_penalty=0.0, # 1.0 
                                                                           num_return_sequences=1, 
                                                                           length_penalty=2.0,
                                                                           no_repeat_ngram_size=3,
                                                                           early_stopping=True)

        # all special tokens including will be removed
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch["pred"] = output_str
        return batch

 
    test_data = datasets.Dataset.from_pandas(reports)
    results = test_data.map(generate_summary, batched=True, batch_size=8)
    pred_str = results["pred"]
    reports['AI_impression'] = pred_str
    
    save_pred_dir = f'{savename}_pred'
    os.makedirs(save_pred_dir, exist_ok=True)
    reports.to_excel(f'{save_pred_dir}/test_{test_epoch}.xlsx', index=False)


if __name__ == '__main__':
    # Testing    
    df = pd.read_excel('./archive/test.xlsx')
    df['impressions'] = df['impressions'].apply(lambda x: x.replace('\n',' '))
    df['findings_info'] = df['findings_info'].apply(lambda x: x.replace('\n',' '))
    print('Start testing')

    # PEGASUS
    test_epoch = 10
    predict_text(df, model_template=f'models/pegasus-large/epoch_{test_epoch}', savename='pegasus-large', test_epoch=test_epoch)