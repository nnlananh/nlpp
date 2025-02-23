from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset("squad")
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = examples["question"]
    targets = examples["answers"]["text"]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=175, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_data = train_dataset.map(preprocess_function, batched=True)
val_data = train_dataset.map(preprocess_function, batched=True)

print(train_data)