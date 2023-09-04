import os 
os.environ['CURL_CA_BUNDLE'] = ''
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install deepspeed xformers==0.0.16") 
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install ninja accelerate --upgrade")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install datasets evaluate")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install -U transformers huggingface-hub tokenizers triton")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install requests==2.27.1 numpy==1.20.3")

import pandas as pd
from transformers import *
import nltk
nltk.download('punkt')
import datasets

def preprocess(df_train, text_column, summary_column, model_id, encoder_max_length, decoder_max_length, save_dataset_path):
    #Select part of data we want to keep
    df_val = df_train.sample(2000, random_state=42)
    df_train = df_train.drop(df_val.index)
    df_train = df_train[[text_column, summary_column]].copy()
    df_val = df_val[[text_column, summary_column]].copy()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        inputs = tokenizer(batch[text_column], padding="max_length", truncation=True, max_length=encoder_max_length)
        outputs = tokenizer(batch[summary_column], padding="max_length", truncation=True, max_length=decoder_max_length)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

        return batch

    train_data = datasets.Dataset.from_pandas(df_train)
    train_data = train_data.map(
        #lambda x: process_data_to_model_inputs(x, tokenizer, encoder_max_length, decoder_max_length) , 
        process_data_to_model_inputs,
        batched=True, 
        batch_size=32)
    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"])
    
    val_data = datasets.Dataset.from_pandas(df_val)
    val_data = val_data.map(
        process_data_to_model_inputs, 
        batched=True, 
        batch_size=32)
    val_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"])
    
    train_data.save_to_disk(os.path.join(save_dataset_path, "train"))
    val_data.save_to_disk(os.path.join(save_dataset_path, "eval"))



if __name__ == "__main__":
    # Get data
    df_train = pd.read_excel('./archive/train.xlsx')
    # clean data
    df_train['impressions'] = df_train['impressions'].apply(lambda x: x.replace('\n', ' '))
    df_train['findings_info'] = df_train['findings_info'].apply(lambda x: x.replace('\n', ' '))  

    text_column = "findings_info" # column of input text is
    summary_column = "impressions" # column of the output text 
    model_id = "google/pegasus-large" # Hugging Face Model Id
    encoder_max_length = 1024 # max length of the input text
    decoder_max_length = 512 # max length of the output text
    lr = 4e-4 # learning rate
    save_dataset_path = "data" # local path to save processed dataset
    # tokenize and save training/validation datasets
    preprocess(df_train, text_column, summary_column, model_id, encoder_max_length, decoder_max_length, save_dataset_path)

    # if 2048, you should reset rerun the preprocess function
    os.system(f'deepspeed --num_gpus=2 run_seq2seq_deepspeed.py \
    --model_id {model_id} \
    --dataset_path {save_dataset_path} \
    --epochs 15 \
    --per_device_train_batch_size 16 \
    --max_position_embedding {encoder_max_length} \
    --generation_max_length {decoder_max_length} \
    --lr {lr} \
    --bf16 False \
    --master_port=25641 \
    --deepspeed configs/ds_config.json')  
    # BF16 half precision training: ds_config_bf16.json
    # FP32 training: ds_config.json
