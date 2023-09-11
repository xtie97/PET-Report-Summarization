# Fine-tuning Large Language Models (LLMs) for PET Report Summarization :bookmark_tabs:

This repository contains the code for the paper [**Automatic Personalized Impression Generation for PET Reports Using Large Language Models**](#link-to-paper) (under review). 

We shared three implementation methods in this repository: 
- [**fastAI Implementation**](https://github.com/xtie97/PET-Report-Summarization/tree/main/fastAI): simple and easy to use
- [**Non-trainer Implementation**](https://github.com/xtie97/PET-Report-Summarization/tree/main/nontrainer): more flexible
- [**Trainer (with deepspeed) Implementation**](https://github.com/xtie97/PET-Report-Summarization/tree/main/deepspeed): reduce memory usage and accelerate training

## 🚀 Getting Started

We already released our model weights in [**Hugging Face**](https://huggingface.co/xtie/PEGASUS-PET-impression). To generate the impressions, run the following code:

```bash
finetuned_model = "xtie/PEGASUS-PET-impression"
tokenizer = AutoTokenizer.from_pretrained(finetuned_model) 
model = AutoModelForSeq2SeqLM.from_pretrained(finetuned_model, ignore_mismatched_sizes=True).eval()

findings_info =
"""
Description: PET CT WHOLE BODY
Radiologist: James
Findings:
Head/Neck: xxx Chest: xxx Abdomen/Pelvis: xxx Extremities/Musculoskeletal: xxx
Indication:
The patient is a 60-year old male with a history of xxx
"""

inputs = tokenizer(findings_info.replace('\n', ' '),
                  padding="max_length",
                  truncation=True,
                  max_length=1024,
                  return_tensors="pt")
input_ids = inputs.input_ids.to("cuda")
attention_mask = inputs.attention_mask.to("cuda")
outputs = model.generate(input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=512, 
                        num_beam_groups=1,
                        num_beams=4, 
                        do_sample=False,
                        diversity_penalty=0.0,
                        num_return_sequences=1, 
                        length_penalty=2.0,
                        no_repeat_ngram_size=3,
                        early_stopping=True
                        )
# get the generated impressions
output_str = tokenizer.decode(outputs,
                              skip_special_tokens=True)

```

## 📚 Citation


_(You can provide a BibTeX citation or other format here.)_


