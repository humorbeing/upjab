from datasets import load_dataset ## Hugging Face Datasets
## Load Dataset from Hugging Face Datasets
emotions = load_dataset("emotion") 
# emotions = load_dataset("csv", data_files = "my_file.csv")
print(emotions)
train_ds = emotions["train"] ## Dataset class, which acts like np.array or list.
# print(train_ds)
# print(len(train_ds))
# print(train_ds[0])
# print(train_ds.features)
# print(train_ds["text"][:5])


import pandas as pd
emotions.set_format(type="pandas")
emotions["train"] = emotions["train"].select(range(1600))
emotions["validation"] = emotions["validation"].select(range(200))
emotions["test"] = emotions["test"].select(range(200))
df = emotions["train"][:]
# print(df.head())
def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)
df["label_name"] = df["label"].apply(label_int2str)
print(df.head())
print(df["label_name"].value_counts(ascending=True))


## Text to tokens
from transformers import AutoTokenizer
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
print(tokenizer.vocab_size, tokenizer.model_max_length)
# Test
text = "Tokenizing text is a core task of NLP."
encoded_text = tokenizer(text)
print(encoded_text)
tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)
print(tokenizer.convert_tokens_to_string(tokens))

## Tokenizing all data
def tokenize(batch):
    tokens = tokenizer(batch["text"].tolist(), padding=True, truncation=True)
    tokens["label"] = batch["label"]
    return tokens
# test
print(tokenize(emotions["train"][:2])) ## 0 means [PAD].
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None) 
# batched=True -> map for each sample, batch_size=None -> map all data at once without batch
print(emotions_encoded["train"].column_names) # ['label', 'input_ids', 'attention_mask']


## Train text classifier: 1. Transfomer for feature extraction, Just train linear classifier
from transformers import AutoModel
# TFAutoModel: TFAutoModel.from_pretrained(model_ckpt, from_pt=True)
import torch
model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # torch.device("cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)
# test
text = "this is a test"
inputs = tokenizer(text, return_tensors="pt")
print(inputs["input_ids"].size()) # batch_size * n_tokens
inputs = {k:v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(input_ids = inputs["input_ids"], attention_mask = inputs["attention_mask"])
print(outputs)
print(outputs.last_hidden_state.size()) # batch_size * n_tokens * hidden_dimension
outputs.last_hidden_state[:, 0] # Use hidden state associated with [CLS] token as input feature



## Feature extraction for the whole dataset
def extract_hidden_states(batch):
    inputs = {k:v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        last_hidden_state = model(input_ids = inputs["input_ids"], attention_mask = 
inputs["attention_mask"]).last_hidden_state
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True, batch_size=None) 
## If you are running out of memory, set batch_size.
print(emotions_hidden["train"].column_names) ## ['label', 'input_ids', 'attention_mask', 'hidden_state']

# Make feature matrix
import numpy as np
X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])
print(X_train.shape, X_valid.shape)
## Train a simple classifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
lr_clf = LogisticRegression(max_iter = 3000)
lr_clf.fit(X_train, y_train)
print(lr_clf.score(X_valid, y_valid))
dummy_clf = DummyClassifier(strategy="most_frequent") # “uniform” for random guess
dummy_clf.fit(X_train, y_train)
print(dummy_clf.score(X_valid, y_valid))

# 2. fine tuning Transformer
from transformers import AutoModelForSequenceClassification
import torch
num_labels = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device)
from sklearn.metrics import accuracy_score, f1_score
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average = "weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}




from transformers import Trainer, TrainingArguments
batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned_emotion"
# Setting for training
training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=5,
    learning_rate=1e-4,
    lr_scheduler_type="constant",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=False,
    save_strategy="epoch",
    load_best_model_at_end=True,
    log_level="error"
    )


trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=emotions_encoded["train"],
    eval_dataset=emotions_encoded["validation"],
    tokenizer=tokenizer)
trainer.train()
preds_output = trainer.predict(emotions_encoded["validation"])
print(preds_output.metrics)



from torch.nn.functional import cross_entropy
def forward_pass_with_label(batch):
# Place all input tensors on the same device as the model
    inputs = {k:v.to(device) for k,v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        output = model(input_ids = inputs["input_ids"], attention_mask = inputs["attention_mask"])
    pred_label = torch.argmax(output.logits, axis=-1)
    loss = cross_entropy(output.logits, batch["label"].to(device), reduction="none")
    # Place outputs on CPU for compatibility with other dataset columns
    return {"loss": loss.cpu().numpy(),
    "predicted_label": pred_label.cpu().numpy()}
# Convert our dataset back to PyTorch tensors
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])


# Compute loss values
emotions_encoded["validation"] = emotions_encoded["validation"].map(forward_pass_with_label, 
batched=True, batch_size=16)
emotions_encoded.set_format("pandas")
# cols = ["text", "label", "predicted_label", "loss"]
cols = ["label", "predicted_label", "loss"]
df_test = emotions_encoded["validation"][:][cols]
df_test["label"] = df_test["label"].apply(label_int2str)
df_test["predicted_label"] = df_test["predicted_label"].apply(label_int2str)
# Set all columns and rows to not be truncated
pd.set_option('display.max_columns', None) # Print all rows
pd.set_option('display.max_rows', None) # Print all columns
pd.set_option('display.expand_frame_repr', False) # Print without scrolling sideways
pd.set_option('display.max_colwidth', None) # Print all long strings
print(df_test.sort_values("loss", ascending=False).head(10)) # Top 10 test data with the largest errors
print(df_test.sort_values("loss", ascending=True).head(10)) # Top 10 test data with the smallest errors






print('done')