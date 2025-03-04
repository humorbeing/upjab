

from datasets import load_dataset
# LoadCNN/DailyMaildataset
dataset= load_dataset("cnn_dailymail", "3.0.0")
print(f"Features: {dataset['train'].column_names}")# ==> ['article', 'highlights', 'id’]
# Checkthedata
sample= dataset["train"][0]
print(f"""Article(excerptof 500 characters, totallength: {len(sample["article"])}):""")
print(sample["article"][:500])
print(f'\nSummary(length: {len(sample["highlights"])}):')
print(sample["highlights"])
sample_text= dataset["train"][0]["article"][:2000]
summaries= {}



# NLTK Tokenizer: Test
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt_tab") ## ifnotworking, trywith"punkt"
string= "The U.S. areacountry. The U.N. isanorganization."
print(sent_tokenize(string))


# Baseline1: Three_sentence_summary-takethefirstthreesentences
def three_sentence_summary(text):
    return"\n".join(sent_tokenize(text)[:3])

summaries["baseline"] = three_sentence_summary(sample_text)



# Method 1: GPT
from transformers import pipeline, set_seed
set_seed(42)
pipe = pipeline("text-generation", model="gpt2")  # "gpt2-xl' for large version
gpt2_query = sample_text + "\nTL;DR:\n"
pipe_out = pipe(gpt2_query, max_length=512, clean_up_tokenization_spaces=True)
summaries["gpt2"] = "\n".join(sent_tokenize(pipe_out[0]["generated_text"][len(gpt2_query):]))


# Method 2: T5
pipe = pipeline("summarization", model="t5-small") # "t5-large" for large version
pipe_out = pipe(sample_text)
summaries["t5"] = "\n".join(sent_tokenize(pipe_out[0]["summary_text"]))
# Method 3: BART
pipe = pipeline("summarization", model="ainize/bart-base-cnn") # "facebook/bart-large-cnn" forlarge version
pipe_out = pipe(sample_text) 
summaries["bart"] = "\n".join(sent_tokenize(pipe_out[0]["summary_text"]))



# Method 4: PEGASUS
pipe = pipeline("summarization", model="sshleifer/distill-pegasus-cnn-16-4") # "google/pegasus-cnn_dailymail" for large version
pipe_out = pipe(sample_text)
summaries["pegasus"] = pipe_out[0]["summary_text"].replace(" .<n>", ".\n")



import evaluate
import pandas as pd
rouge_metric= evaluate.load("rouge")
reference= dataset["train"][0]["highlights"]
records= []
rouge_names= ["rouge1", "rouge2", "rougeL"]
for model_name in summaries:
    rouge_metric.add(prediction=summaries[model_name], reference=reference)
    score= rouge_metric.compute()
    rouge_dict= {rn: score[rn] for rn in rouge_names}     
    records.append(rouge_dict)
print(pd.DataFrame.from_records(records, index=summaries.keys()))



from datasets import load_dataset
from transformers import pipeline, set_seed
set_seed(42)
# Load and check dataset
dataset_samsum = load_dataset("samsum", trust_remote_code=True)
split_lengths = [len(dataset_samsum[split]) for split in dataset_samsum]
print(f"Split lengths: {split_lengths}")
print(f"Features: {dataset_samsum['train'].column_names}")
print("\nDialogue:")
print(dataset_samsum["test"][0]["dialogue"])
print("\nSummary:")
print(dataset_samsum["test"][0]["summary"])




# TestusingPEGASUS
pipe= pipeline("summarization", model="sshleifer/distill-pegasus-cnn-16-4") # "google/pegasus-cnn_dailymail" forlargeversion
pipe_out= pipe(dataset_samsum["test"][0]["dialogue"], max_length=60)
print("Summary:")
print(pipe_out[0]["summary_text"].replace(" .<n>", ".\n"))



from transformers import AutoTokenizer
from transformers import MarianTokenizer, AutoModelForSeq2SeqLM
model_ckpt = "sshleifer/distill-pegasus-cnn-16-4"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
dataset_samsum["train"] = dataset_samsum["train"].select(range(len(dataset_samsum["train"]) // 100))
dataset_samsum["validation"] = dataset_samsum["validation"].select(range(len(dataset_samsum["validation"]) // 100))
dataset_samsum["test"] = dataset_samsum["test"].select(range(len(dataset_samsum["test"]) // 100))


def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch["dialogue"], max_length=1024, truncation=True)
    target_encodings = tokenizer(text_target=example_batch["summary"], max_length=128, truncation=True)
    return {"input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"]}
dataset_samsum_pt = dataset_samsum.map(convert_examples_to_features, batched=True)
columns = ["input_ids", "labels", "attention_mask"]
dataset_samsum_pt.set_format(type="torch", columns=columns)


from transformers import DataCollatorForSeq2Seq
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
# Trainer
from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
    output_dir='pegasus-samsum', num_train_epochs=1, warmup_steps=500,
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    weight_decay=0.01, logging_steps=10, push_to_hub=False,
    evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
    gradient_accumulation_steps=16)
trainer = Trainer(model=model, args=training_args,
            tokenizer=tokenizer, data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["train"],
            eval_dataset=dataset_samsum_pt["validation"])
trainer.train()



sample_text= dataset_samsum["test"][0]["dialogue"]
reference= dataset_samsum["test"][0]["summary"]
pipe= pipeline("summarization", model=model, tokenizer=tokenizer)
print("Dialogue:")
print(sample_text)
print("\nReferenceSummary:")
print(reference)
print("\nModelSummary:")
print(pipe(sample_text, length_penalty=0.8, num_beams=8, max_length=128)[0]["summary_text"])



print('done')