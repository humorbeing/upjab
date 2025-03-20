
from datasets import load_dataset
from collections import defaultdict
from datasets import DatasetDict
langs = ["de", "fr", "it", "en"]
fracs = [0.629, 0.229, 0.084, 0.059]
panx_ch = defaultdict(DatasetDict)
for lang, frac in zip(langs, fracs):
    # Load monolingual corpus
    ds = load_dataset("xtreme", name=f"PAN-X.{lang}")
    # Shuffle and downsample each split according to spoken proportion
    for split in ds:
        panx_ch[lang][split] = (
            ds[split]
            .shuffle(seed=0)
            .select(range(int(frac * ds[split].num_rows) // 10)))



import pandas as pd
# Number of training examples for each language
print(pd.DataFrame({lang: [panx_ch[lang]["train"].num_rows] for lang in langs}, index=["Number of training examples"]))
# test
element = panx_ch["de"]["train"][0]
for key, value in element.items():
    print(f"{key}: {value}")
# Class labels
tags = panx_ch["de"]["train"].features["ner_tags"].feature
print(tags)
# map integer labels to str labels
def create_tag_names(batch):
    return {"ner_tags_str": [tags.int2str(idx) for idx in batch["ner_tags"]]}
panx_de = panx_ch["de"].map(create_tag_names)
# test: check labels for an instance
de_example = panx_de["train"][0]
pd.set_option('display.max_columns', None)  # Print all columns
print(pd.DataFrame(data=[de_example["tokens"], de_example["ner_tags_str"]], index=['Tokens', 'Tags']))


## Configuration
from transformers import XLMRobertaForTokenClassification
xlmr_model_name = "xlm-roberta-base"
index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
tag2index = {tag: idx for idx, tag in enumerate(tags.names)}


from transformers import AutoConfig
xlmr_config = AutoConfig.from_pretrained(xlmr_model_name,
    num_labels=tags.num_classes,
    id2label=index2tag, label2id=tag2index)

# Load pretrained model
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.device("cpu")
xlmr_model = ((XLMRobertaForTokenClassification
    .from_pretrained(xlmr_model_name, config=xlmr_config))
    .to(device))




from transformers import AutoTokenizer
xlmr_tokenizer = AutoTokenizer.from_pretrained(xlmr_model_name)
def tokenize_and_align_labels(examples):
    tokenized_inputs = xlmr_tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for idx, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs
def encode_panx_dataset(corpus):
    return corpus.map(tokenize_and_align_labels, batched=True, remove_columns=['langs', 'ner_tags', 'tokens'])
panx_de_encoded = encode_panx_dataset(panx_ch["de"])





from seqeval.metrics import classification_report
y_true = [["O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O"],       ["B-PER", "I-PER", "O"]]
y_pred = [["O", "O", "B-MISC", "I-MISC", "I-MISC", "I-MISC", "O"],  ["B-PER", "I-PER", "O"]]
print(classification_report(y_true, y_pred))





import numpy as np
def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    labels_list, preds_list = [], []
    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        for seq_idx in range(seq_len):
            # Ignore label IDs = -100
            if label_ids[batch_idx, seq_idx] != -100:
                example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
                example_preds.append(index2tag[preds[batch_idx][seq_idx]])
        labels_list.append(example_labels)
        preds_list.append(example_preds)
    return preds_list, labels_list




## Fine-tuningpretrainedmodel
# TrainingArguments
from transformers import TrainingArguments
num_epochs= 3
batch_size= 24
logging_steps= len(panx_de_encoded["train"]) // batch_size
model_name= f"{xlmr_model_name}-finetuned-panx-de"
training_args= TrainingArguments(
    output_dir=model_name, log_level="error", num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size, evaluation_strategy="epoch",
    save_steps=1e6, weight_decay=0.01, disable_tqdm=False,
    logging_steps=logging_steps, push_to_hub=False, fp16=True)





# Metrics
from seqeval.metrics import f1_score
def compute_metrics(eval_pred):
    y_pred, y_true = align_predictions(eval_pred.predictions, eval_pred.label_ids)
    return {"f1": f1_score(y_true, y_pred)}
# Padding for labels
from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(xlmr_tokenizer)





# Trainthemodel
def model_init():
    return(XLMRobertaForTokenClassification
        .from_pretrained(xlmr_model_name, config=xlmr_config)
        .to(device))
from transformers import Trainer
trainer= Trainer(model_init=model_init, args=training_args,
    data_collator=data_collator, compute_metrics=compute_metrics,
    train_dataset=panx_de_encoded["train"],
    eval_dataset=panx_de_encoded["validation"],
    tokenizer=xlmr_tokenizer)
trainer.train()





## Cross-lingual Transfer Performance
def get_f1_score(trainer, dataset):
    return trainer.predict(dataset).metrics["test_f1"]
# 1. de to de
f1_scores = defaultdict(dict)
f1_scores["de"]["de"] = get_f1_score(trainer, panx_de_encoded["test"])
print(f"F1-score of [de] model on [de] dataset: {f1_scores['de']['de']:.3f}")





# 2. detoothers
def evaluate_lang_performance(lang, trainer):
    panx_ds= encode_panx_dataset(panx_ch[lang])
    return get_f1_score(trainer, panx_ds["test"])
f1_scores["de"]["fr"] = evaluate_lang_performance("fr", trainer)
print(f"F1-score of [de] modelon[fr] dataset: {f1_scores['de']['fr']:.3f}")
f1_scores["de"]["it"] = evaluate_lang_performance("it", trainer)
print(f"F1-score of [de] modelon[it] dataset: {f1_scores['de']['it']:.3f}")
f1_scores["de"]["en"] = evaluate_lang_performance("en", trainer)
print(f"F1-score of [de] modelon[en] dataset: {f1_scores['de']['en']:.3f}")


print('done')