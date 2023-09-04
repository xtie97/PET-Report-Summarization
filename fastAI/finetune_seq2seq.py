import os 
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install bert-score fastai==2.7.11") 
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install scipy uvicorn gunicorn==19.9.0 fastapi uvloop httptools")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install ohmeow-blurr==1.0.5")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install -U huggingface-hub transformers tokenizers")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install requests==2.27.1")

# transformers==4.26.1  
import pandas as pd
from fastai.text.all import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from blurr.text.data.all import *
from blurr.text.modeling.all import *
import nltk
nltk.download('punkt')

def finetune(reports, pretrained_model_name, savename, lr, batchsize, batchsize_per_device, finetune_nepoch):
    #Select part of data we want to keep
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, use_fast=True)
    model_template = "./template/my-{}".format(pretrained_model_name.split('/')[-1])
    # save model and tokenizer in a local folder 
    model.save_pretrained(model_template)
    tokenizer.save_pretrained(model_template) 

    # wrap the model and tokenizer for fastai
    hf_arch, hf_config, hf_tokenizer, hf_model = get_hf_objects(pretrained_model_name, 
                                                                model_cls=model)
   
    #Create mini-batch and define parameters
    batch_tokenize_tfm = Seq2SeqBatchTokenizeTransform(hf_arch, hf_config, hf_tokenizer, hf_model, 
        task='summarization',
        max_length=1024, # for BART, PEGASUS, T5
        max_target_length=512, 
        padding=True, truncation=True)
    
    #Prepare data for training
    blocks = (Seq2SeqTextBlock(batch_tokenize_tfm=batch_tokenize_tfm), noop)
    dblock = DataBlock(blocks=blocks, get_x=ColReader('findings_info'), get_y=ColReader('impressions'), 
                        splitter=RandomSplitter(valid_pct=0.06, seed=42)) # valid_sample=2000 

    # create batch of samples
    dls = dblock.dataloaders(reports, batch_size=batchsize_per_device)
    #b = dls.one_batch()
    #print(len(b), b[0]["input_ids"].shape, b[1].shape) 
    
    # define metrics for monitoring training
    seq2seq_metrics = {
        "rouge": {
            "compute_kwargs": {"rouge_types": ["rouge1", "rouge2", "rougeL", "rougeLsum"], "use_stemmer": True},
            "returns": ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        },
    }

    model = BaseModelWrapper(hf_model).to('cuda')
    learn_cbs = [BaseModelCallback]
    
    learn = Learner(
        dls,
        model,
        opt_func=partial(Adam),
        loss_func=CrossEntropyLossFlat(), # standard cross-entropy loss for LM
        cbs=learn_cbs,
        model_dir='models', # where to save the model
    )

    # learn = learn.to_native_fp16() #.to_fp16()
    learn.create_opt() 
    learn.unfreeze() 
    init_lr = last_lr = lr # or larger lr in the last few layers and smaller lr in the first few layers
    print(learn.summary())
    # pick suitable learning rate
    #print(learn.lr_find(suggest_funcs=[steep, valley, slide])) # typically, we pick either valley or slide or any value in between
    # example: steep=0.00013182566908653826, valley=0.00013182566908653826, slide=0.0003311311302240938>  -> 0.0002 is a good choice for PEGASUS
    
    fit_cbs = [Seq2SeqMetricsCallback(custom_metrics=seq2seq_metrics), 
            SaveModelCallback(fname=f'{savename}_{lr}', every_epoch=True),
            GradientAccumulation(n_acc=batchsize)]
    learn.fit(finetune_nepoch, slice(init_lr, last_lr), cbs=fit_cbs)
    

if __name__ == "__main__":
    #Get data
    df_train = pd.read_excel('./archive/train.xlsx')
    df_train['impressions'] = df_train['impressions'].apply(lambda x: x.replace('\n',' '))
    df_train['findings_info'] = df_train['findings_info'].apply(lambda x: x.replace('\n',' '))

    # PEGASUS
    finetune(df_train, "google/pegasus-large", "pegasus_large", lr=4e-4, batchsize=32, batchsize_per_device=4, finetune_nepoch=15)

    # BART
    finetune(df_train, "facebook/bart-large", "bart_large", lr=5e-5, batchsize=32, batchsize_per_device=4, finetune_nepoch=15)

    # T5V1
    finetune(df_train, "google/t5-v1_1-large", "t5_large", lr=2e-4, batchsize=32, batchsize_per_device=4, finetune_nepoch=15)
