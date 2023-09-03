import os 
os.environ["CURL_CA_BUNDLE"]=""
os.environ["PYTHONWARNINGS"]="ignore:Unverified HTTPS request"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.system('pip install -U tqdm datasets evaluate')
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install bert-score fastai==2.7.11") 
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install scipy uvicorn gunicorn==19.9.0 fastapi uvloop httptools")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install ohmeow-blurr==1.0.5")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install huggingface-hub --upgrade")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install -U transformers")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install requests==2.27.1")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install -U tokenizers")

from transformers import AutoTokenizer, EncoderDecoderModel, EncoderDecoderConfig
#from seq2seq_trainer import Seq2SeqTrainer
import pandas as pd 
import pandas as pd
from fastai.text.all import *
from transformers import *
from blurr.text.data.all import *
from blurr.text.modeling.all import *
import nltk
nltk.download('punkt')

def finetune(report, encoder_name, decoder_name, savename, lr, batchsize, batchsize_per_device, finetune_nepoch):
    
    # Load model and tokenizer, create a seq2seq model by loading pretrained encoder and decoder models
    # cross attention layers are randomly initialized
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder_name, decoder_name)
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    
    encoder_max_length=1024
    decoder_max_length=512
    # Load tokenizer / set special tokens
    tokenizer.pad_token_id = 0 # padding token # from BERT config 
    tokenizer.bos_token_id = 1 # start of sequence
    tokenizer.eos_token_id = 2 # end of sequence
    tokenizer.sep_token_id = 2 # seperator token

    model.config.decoder_start_token_id = tokenizer.bos_token_id 
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.sep_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.max_position_embeddings = encoder_max_length

    model.config.decoder.add_cross_attention = True  
    model.config.decoder.is_decoder = True
    # for beam search / set decoder parameters 
    # model.config.vocab_size = model.config.decoder.vocab_size # 50265
    # generation config
    model.config.max_length = decoder_max_length
    model.config.min_length = 0
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    model_template = "./template/my-{}-{}".format(encoder_name.split('/')[-1], decoder_name.split('/')[-1])
    model.save_pretrained(model_template)
    tokenizer.save_pretrained(model_template) 
    
    # loading model and config from pretrained folder
    model = EncoderDecoderModel.from_pretrained(model_template, ignore_mismatched_sizes=True)
    arch, config, tokenizer, model = get_hf_objects(model_template, model_cls=model)
    print(config)
    
    # Load dataset & Process data
    batch_tokenize_tfm = Seq2SeqBatchTokenizeTransform(arch, config, tokenizer, model, 
        task='summarization',
        max_length=encoder_max_length, 
        max_target_length=decoder_max_length, 
        padding=True, truncation=True)

    #Prepare data for training
    blocks = (Seq2SeqTextBlock(batch_tokenize_tfm=batch_tokenize_tfm), noop)
    dblock = DataBlock(blocks=blocks, get_x=ColReader('findings_info'), get_y=ColReader('impressions'), 
                        splitter=RandomSplitter(valid_pct=0.06, seed=42)) # valid_sample=2000 

    dls = dblock.dataloaders(report, batch_size=batchsize_per_device)
    #b = dls.one_batch()
    #print(len(b), b[0]["input_ids"].shape, b[1].shape) 

    # define metrics for monitoring training
    seq2seq_metrics = {
        "rouge": {
            "compute_kwargs": {"rouge_types": ["rouge1", "rouge2", "rougeL", "rougeLsum"], "use_stemmer": True},
            "returns": ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        },
    }

    model = BaseModelWrapper(model)
    learn_cbs = [BaseModelCallback]
    
    learn = Learner(
        dls,
        model,
        opt_func=partial(Adam),
        loss_func=CrossEntropyLossFlat(),
        cbs=learn_cbs,
        model_dir='models', # where to save the model
    )

    # learn = learn.to_native_fp16() #.to_fp16()
    learn.create_opt() 
    learn.unfreeze() 
    init_lr = last_lr = lr # or larger lr in the last few layers and smaller lr in the first few layers
    print(learn.summary())

    fit_cbs = [Seq2SeqMetricsCallback(custom_metrics=seq2seq_metrics), 
            SaveModelCallback(fname=f'{savename}_{lr}', every_epoch=True),
            GradientAccumulation(n_acc=batchsize)]
    learn.fit(finetune_nepoch, slice(init_lr, last_lr), cbs=fit_cbs)


if __name__ == "__main__":
    #Get data
    df_train = pd.read_excel('./archive/train.xlsx')
    df_train['impressions'] = df_train['impressions'].apply(lambda x: x.replace('\n', ' '))
    df_train['findings_info'] = df_train['findings_info'].apply(lambda x: x.replace('\n', ' '))

    # BERT2BERT (Clinical-longformer2Roberta)    
    finetune(df_train, 'yikuan8/Clinical-Longformer', 'roberta-base', 'cli_longformer_roberta', lr=1e-4, batchsize=32, batchsize_per_device=4, finetune_nepoch=15)

