import os 
os.environ['CURL_CA_BUNDLE'] = ''
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install deepspeed xformers==0.0.16") 
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install ninja accelerate --upgrade")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install datasets evaluate")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install -U transformers huggingface-hub tokenizers triton")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install requests==2.27.1 numpy==1.20.3")

import pandas as pd
from tqdm import tqdm 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import pandas as pd
import deepspeed
# Command: deepspeed --num_gpus=2 predict_XL.py

def predict_text(reports, model_template, savename, test_epoch):
    #Import the pretrained model
    tokenizer = AutoTokenizer.from_pretrained(model_template.split("/")[0], padding_side="left", truncation_side="right")
    tokenizer.pad_token = tokenizer.eos_token  
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_template, pad_token_id=tokenizer.pad_token_id).eval()     
    ''' # deepspeed inference for faster speed
    num_GPU = torch.cuda.device_count()
    ds_model = deepspeed.init_inference(
        model=model,      # Transformers models
        mp_size=num_GPU,        # Number of GPU
        max_out_tokens=1024, # max output length
        dtype=torch.float32, # torch.half: dtype of the weights (fp16)
        # injection_policy={"BertLayer" : HFBertLayerPolicy}, # replace BertLayer with DS HFBertLayerPolicy
        replace_method="auto", # Lets DS autmatically identify the layer to replace
        replace_with_kernel_inject=True, # replace the model with the kernel injector
        )
    num_beams = 1 
    batch_size = 4 
    savename = savename + "_full_deepspeed"  
    '''
    ds_model = model.to("cuda") # or ds_model = model.half().to("cuda") # half precision inference for faster speed and less memory
    num_beams = 4 
    batch_size = 1
    savename = savename + "_full"

    decoder_max_length = 256
    encoder_max_length = 1024 - decoder_max_length # 2048 is the max length in OPT 
    encoded_prefix = tokenizer('\n\n### Response:\n', add_special_tokens=False, truncation=False, padding=False, return_tensors="pt").input_ids 
    # duplicate the encoded_prefix along the batch dimension
    encoded_prefix = encoded_prefix.expand(batch_size, -1)
    decoder_max_length -= encoded_prefix.shape[-1] # 3
    
    # map data correctly
    def generate_summary(batch):
        text = batch['input']
        encoded_prompt = tokenizer(text, add_special_tokens=False, truncation=True, padding="max_length", 
                                                    max_length=encoder_max_length, return_tensors="pt").input_ids 
        encoded_prompt = torch.cat([encoded_prompt, encoded_prefix], dim=-1) 
        encoded_prompt = encoded_prompt.to("cuda") 
        
        # prediction
        outputs = ds_model.generate(input_ids=encoded_prompt,
                                    max_new_tokens=decoder_max_length,
                                    no_repeat_ngram_size=3, 
                                    temperature=1.0,
                                    num_beam_groups=1,
                                    num_beams=num_beams, 
                                    do_sample=False,
                                    diversity_penalty=0.0, # 1.0 
                                    length_penalty=2.0,
                                    early_stopping=True)
        # all special tokens including will be removed
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch["pred"] = output_str  
        return batch

    test_data = datasets.Dataset.from_pandas(reports)
    results = test_data.map(generate_summary, batched=True, batch_size=batch_size)
    pred_str = results["pred"]
    reports['AI_impression_all'] = pred_str

    for ii, pred in enumerate(pred_str):
        pred_str[ii] = pred.split("\n\n### Response:\n")[-1]  
    reports['AI_impression'] = pred_str

    save_pred_dir = f'{savename}_pred'
    os.makedirs(save_pred_dir, exist_ok=True)
    reports.to_excel(f'{save_pred_dir}/test_{test_epoch}.xlsx', index=False)


if __name__ == '__main__':
    # Testing    
    df = pd.read_excel('./archive/test.xlsx')

    input_list = []
    # Iterate through the rows and create a dictionary for each row
    prefix = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request." 
    for _, row in tqdm(df.iterrows()):
        instruction = "Derive the impression from the given {} report for {}.".format(row['Study Description'], row['Reading Radiologist']) 
        input = "{}\n{}".format(row['findings'].replace('\n', ' '), row['merged_information'].replace('\n', ' '))
        res = prefix + f"\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}" # \n\n### Response:\n 
        input_list.append(res)
    df['input'] = input_list
  
    print('Start testing')
    model_template = f"my-gpt2-xl/checkpoint-13650" 
    predict_text(df, model_template, 'gpt2-xl', test_epoch=13)
