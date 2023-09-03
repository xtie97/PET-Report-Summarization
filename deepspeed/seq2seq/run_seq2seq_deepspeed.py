import os
import argparse
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    set_seed,
)
from datasets import load_from_disk
import urllib3
urllib3.disable_warnings()
import torch
import nltk
nltk.download("punkt", quiet=True)
import evaluate
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments


def parse_arg():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument("--model_id", type=str, default="google/flan-t5-xl", help="Model id to use for training.")
    parser.add_argument("--dataset_path", type=str, default="data", help="Path to the already processed dataset.")
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train for.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size to use for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size to use for testing.")
    parser.add_argument("--max_position_embedding", type=int, default=1024, help="Maximum input sequence length.")
    parser.add_argument("--generation_max_length", type=int, default=512, help="Maximum length to use for generation")
    parser.add_argument("--generation_num_beams", type=int, default=1, help="Number of beams to use for generation.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="To reduce memory usage.")
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    args = parser.parse_known_args()
    return args

def training_function(args):
    # set seed
    set_seed(args.seed)

    # load dataset from disk and tokenizer
    train_dataset = load_from_disk(os.path.join(args.dataset_path, "train"))
    eval_dataset = load_from_disk(os.path.join(args.dataset_path, "eval"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # Save our tokenizer and create model card
    output_dir = args.model_id.split("/")[-1]
    tokenizer.save_pretrained(output_dir)
    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )
    model.config.n_positions = args.max_position_embedding
    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8, max_length=args.max_position_embedding
    )

    # load rouge for validation
    rouge = evaluate.load("rouge")
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # all unnecessary tokens are removed
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rougeL"])["rougeL"]
        print(pred_str)
        return {
            "rougeL": round(rouge_output, 3)
            }
    gradient_accumulation_steps = 32 // args.per_device_train_batch_size // torch.cuda.device_count()  
    
    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        do_train=True,
        do_eval=False, # for eval on validation set
        evaluation_strategy="no", # for eval on validation set "epoch", "no" for not evaluation 
        predict_with_generate=False, # for eval on validation set
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        load_best_model_at_end=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=False,  # T5 overflows with fp16 
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
        # logging & evaluation strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=128,
        save_strategy="epoch",
        save_total_limit=100,
        # push to hub parameters
        report_to="tensorboard",
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics) 
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    # Start training
    trainer.train()


if __name__ == "__main__":
    args, _ = parse_arg()
    training_function(args)
