import os 
os.environ['CURL_CA_BUNDLE'] = ''
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install bert-score fastai==2.7.11") 
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install scipy uvicorn gunicorn==19.9.0 fastapi uvloop httptools")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install ohmeow-blurr==1.0.5")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install huggingface-hub --upgrade")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install -U transformers")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install requests==2.27.1")

import pandas as pd
import torch
from fastai.text.all import *
from transformers import AutoModelForSeq2SeqLM, EncoderDecoderModel
from blurr.text.data.all import *
from blurr.text.modeling.all import *
import datasets

def predict_text(reports, pretrained_model_name, savename, lr, test_epoch):
    if isinstance(pretrained_model_name, list):
        encoder_name, decoder_name = pretrained_model_name[0], pretrained_model_name[1]
        model_template = "./template/my-{}-{}".format(encoder_name.split('/')[-1], decoder_name.split('/')[-1])
        model = EncoderDecoderModel.from_pretrained(model_template, ignore_mismatched_sizes=True)
    else:
        model_template = "./template/my-{}".format(pretrained_model_name.split('/')[-1]) 
        model = AutoModelForSeq2SeqLM.from_pretrained(model_template) 

    arch, config, tokenizer, model = get_hf_objects(model_template, model_cls=model)
    
    #Create mini-batch and define parameters
    encoder_max_length = 1024 
    decoder_max_length = 512
    export_fname = f"{savename}_{lr}_{test_epoch}.pth"
    net_dict = torch.load("models/" + export_fname) 
    net_dict = {k.replace('hf_model.', ''): v for k, v in net_dict.items()}
    model.load_state_dict(net_dict)
    model.to("cuda").eval()
  
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
    reports.to_excel(f'{save_pred_dir}/test_{lr}_{test_epoch}.xlsx', index=False)

if __name__ == '__main__':
    # load test data     
    df = pd.read_excel('./archive/test.xlsx')
    df['impressions'] = df['impressions'].apply(lambda x: x.replace('\n',' '))
    df['findings_info'] = df['findings_info'].apply(lambda x: x.replace('\n',' '))

    print('Start testing')
    # PEGASUS
    predict_text(df, "google/pegasus-large", 'pegasus_large', lr=4e-4, test_epoch=10)

    # BART
    predict_text(df, "facebook/bart-large", "bart_large", lr=5e-5, test_epoch=9)

    # T5V1
    predict_text(df, "google/t5-v1_1-large", "t5_large", lr=2e-4, test_epoch=12)

    # BERT2BERT (Clinical-longformer2Roberta)    
    predict_text(df, ["yikuan8/Clinical-Longformer", "roberta-base"], "cli_longformer_roberta", lr=1e-4, test_epoch=11)



