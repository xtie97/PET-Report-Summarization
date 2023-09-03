import os 
os.environ['CURL_CA_BUNDLE'] = ''
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install deepspeed xformers==0.0.16") 
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install ninja accelerate --upgrade")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install datasets evaluate")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install -U transformers huggingface-hub tokenizers triton")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install requests==2.27.1 numpy==1.20.3")

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datasets
import pandas as pd
import deepspeed
# Command: deepspeed --num_gpus=2 predict_XL.py

def predict_text(reports, model_template, savename, run_deepspeed, run_half, test_epoch):
    #Import the pretrained model
    num_GPU = torch.cuda.device_count()
    tokenizer = AutoTokenizer.from_pretrained(model_template.split('/')[:-1])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_template, ignore_mismatched_sizes=True).eval()
    if run_half:
        # default is fp32 or pass "auto" to torch_dtype
        if run_deepspeed:
            # init deepspeed inference engine
            ds_model = deepspeed.init_inference(
                model=model,      # Transformers models
                mp_size=num_GPU,        # Number of GPU
                dtype=torch.half,       # dtype of the weights (fp16)
                replace_method="auto", # Lets DS autmatically identify the layer to replace
                replace_with_kernel_inject=True, # replace the model with the kernel injector
                )
            num_beams = 1 # deepspeed only supports 1 beam
            batchsize = 32 # 32 CPU cores
            savename = savename + "_half_deepspeed" # This is also the default 
        else:
            ds_model = model.half().to('cuda')
            num_beams = 4 
            batchsize = 8
            savename = savename + "_half"
    else:
        if run_deepspeed:
            # init deepspeed inference engine
            ds_model = deepspeed.init_inference(
                model=model,      # Transformers models
                max_out_tokens=1024, # max number of tokens to generate
                mp_size=num_GPU,        # Number of GPU
                dtype = torch.float32,
                replace_method="auto", # Lets DS autmatically identify the layer to replace
                replace_with_kernel_inject=True, # replace the model with the kernel injector
                )
            num_beams = 1 
            batchsize = 16  
            savename = savename + "_full_deepspeed"
        else:
            ds_model = model.to('cuda') # usually the best 
            num_beams = 4 
            batchsize = 4  
            savename = savename + "_full"

    encoder_max_length = 1024
    decoder_max_length = 512
    # map data correctly
    def generate_summary(batch):
        # Tokenizer will automatically set [BOS] <text> [EOS]
        inputs = tokenizer(batch["findings_info"], padding="max_length", truncation=True, max_length=encoder_max_length, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")
        outputs = ds_model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=decoder_max_length, 
                                                                           num_beam_groups=1,
                                                                           num_beams=num_beams, 
                                                                           do_sample=False,
                                                                           diversity_penalty=0.0,
                                                                           num_return_sequences=1, 
                                                                           length_penalty=2.0,
                                                                           no_repeat_ngram_size=3,
                                                                           early_stopping=True)
       
        # all special tokens including will be removed
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch["pred"] = output_str
        return batch

    test_data = datasets.Dataset.from_pandas(reports)
    results = test_data.map(generate_summary, batched=True, batch_size=batchsize)
    pred_str = results["pred"]
    reports['AI_impression'] = pred_str
    
    save_pred_dir = f'{savename}_pred'
    os.makedirs(save_pred_dir, exist_ok=True)
    reports.to_excel(f'{save_pred_dir}/test_{test_epoch}.xlsx', index=False)


if __name__ == '__main__':
    # Testing    
    df = pd.read_excel('./archive/test.xlsx')
    df['impressions'] = df['impressions'].apply(lambda x: x.replace('\n', ' '))
    df['findings_info'] = df['findings_info'].apply(lambda x: x.replace('\n', ' '))
    
    print('Start testing')
    
    model_template = f"pegasus-large/checkpoint-12600" # load the fine-tuned model in the local folder  
    predict_text(df, model_template, savename='pegasus-large', run_deepspeed=False, run_half=False, test_epoch=12)