import os 
os.environ['CURL_CA_BUNDLE'] = ''
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install deepspeed xformers==0.0.16") 
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install ninja accelerate --upgrade")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install datasets evaluate")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install -U transformers huggingface-hub tokenizers triton")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install requests==2.27.1 numpy==1.20.3")

import pandas as pd
from transformers import AutoTokenizer
import nltk
nltk.download('punkt')
from tqdm import tqdm
from datasets import load_dataset
from utils.prompter import Prompter
import json

# Convert the excel file to a JSON file
def convert_xlsx2json(excel_file, json_file):
    df = pd.read_excel(excel_file, engine='openpyxl')

    # Create a list to store the formatted data
    formatted_data = []

    # Iterate through the rows and create a dictionary for each row

    for _, row in tqdm(df.iterrows()):
        data = {
            "instruction": "Derive the impression from the given {} report for {}.".format(row['Study Description'], row['Reading Radiologist']),
            "input": "{}\n{}".format(row['findings'].replace('\n', ' '), row['merged_information'].replace('\n', ' ')),
            "output": row['impressions'].replace('\n', ' ')
        }
        formatted_data.append(data)

    # Write the formatted data to a JSON file
    with open(json_file, 'w') as f:
        json.dump(formatted_data, f, indent=4)
    
# Preprocess the data
def preprocess(model_id, encoder_max_full, encoder_max_response, save_dataset_path):
    #Select part of data we want to keep
    convert_xlsx2json('archive/train.xlsx', 'archive/train.json')
    convert_xlsx2json('archive/test.xlsx', 'archive/test.json')

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", truncation_side="right") # "left" for batch generation 
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    #print(tokenizer.pad_token, tokenizer.bos_token, tokenizer.eos_token, tokenizer.unk_token)
    #print(tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id)
    
    def tokenize(prompt, encoder_max_length, add_eos_token, padding=False):  # add eos token to the end of the prompt
        # there's probably a way to do this with the tokenizer settings
        result = tokenizer(
            prompt,
            max_length=encoder_max_length,
            truncation=True,
            padding="max_length" if padding else False, 
            return_tensors=None,
        ) 
        # True or 'longest': pad to the longest sequence in the batch (no padding is applied if you only provide a single sequence).
        # 'max_length': pad to a maximum length specified with the argument max_length
        if (result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < encoder_max_length
            and add_eos_token
        ): # add_eos_token = True, 
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    response_prefix = tokenize( '\n\n### Response:\n', encoder_max_length=10, add_eos_token=False, padding=False)
    response_prefix_len = len(response_prefix["input_ids"])  
    
    Prompter_PT = Prompter(verbose=False)
    def generate_and_tokenize_prompt(data_point):
        instruct_prompt = Prompter_PT.generate_prompt(data_point["instruction"], data_point["input"])
        tokenized_response = tokenize( '\n\n### Response:\n' + data_point["output"], encoder_max_length=encoder_max_response, add_eos_token=True, padding=False)
        # add eos_token (2 = bos_token) at the end, not pad the response (it will be padded later)
        user_response_len = len(tokenized_response["input_ids"]) 

        tokenized_user_prompt = tokenize(instruct_prompt, encoder_max_length=encoder_max_full-encoder_max_response, add_eos_token=False, padding=True)
        # not add eos_token at the end, pad the prompt to the max length for the encoder
        user_prompt_len = len(tokenized_user_prompt["input_ids"]) + response_prefix_len

        tokenized_full_prompt = tokenized_user_prompt.copy()
        tokenized_full_prompt['input_ids'] = tokenized_user_prompt['input_ids'] + tokenized_response['input_ids']
        tokenized_full_prompt['attention_mask'] = tokenized_user_prompt['attention_mask'] + tokenized_response['attention_mask']
        tokenized_full_prompt['labels'] = tokenized_user_prompt['labels'] + tokenized_response['labels']

        # not train on inputs, mask out the inputs when computing the loss 
        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]   

        return tokenized_full_prompt

    ### Train data ###
    data = load_dataset("json", data_files='archive/train.json')  
    train_val = data["train"].train_test_split(test_size=0.06, seed=42) # default shuffle=True 

    train_data = (train_val["train"].map(generate_and_tokenize_prompt)) 
    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"])
    train_data.save_to_disk(os.path.join(save_dataset_path, "train"))
    
    val_data = (train_val["test"].map(generate_and_tokenize_prompt))
    val_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_data.save_to_disk(os.path.join(save_dataset_path, "val"))

    ### Test data ###
    data = load_dataset("json", data_files='archive/test.json')
    test_data = (data["train"].map(generate_and_tokenize_prompt))
    test_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_data.save_to_disk(os.path.join(save_dataset_path, "test"))
    

if __name__ == "__main__":
    model_id = "llama-7B"  # 
    encoder_max_full = 2048 # max length of the input text
    encoder_max_response = 512 # max length of the response text 
    lr = 2e-4 # learning rate
    save_dataset_path = f"data" # local path to save processed dataset
    #preprocess(model_id, encoder_max_full, encoder_max_response, save_dataset_path)
     
    # Run the training script
    os.system(f'deepspeed --num_gpus=2 run_casualLM_deepspeed.py \
    --model_id {model_id} \
    --dataset_path {save_dataset_path} \
    --epochs 20 \
    --per_device_train_batch_size 6 \
    --lr {lr} \
    --bf16 True \
    --gradient_checkpointing True \
    --master_port=25690 \
    --deepspeed configs/ds_config_bf16.json')  
    
