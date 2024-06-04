import nltk
import evaluate
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
from datetime import datetime
os.environ["WANDB_DISABLED"] = "true"

class trainer_cls:
   def __init__(self,model_name,df,columns) -> None:
        # Load the tokenizer, model, and data collator
      self.results=[]
      self.columns=columns
      self.data_df=df
      self.MODEL_NAME = model_name  #"google/flan-t5-base"
      self.tokenizer = T5Tokenizer.from_pretrained(self.MODEL_NAME)
      self.model = T5ForConditionalGeneration.from_pretrained(self.MODEL_NAME)
      self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
      nltk.download("punkt", quiet=True)
      self.metric = evaluate.load("rouge")
   def preprocess_function(self,examples):
         """Add prefix to the sentences, tokenize the text, and set the labels"""
         # The "inputs" are the tokenized answer:
         inputs = [self.prefix + doc for doc in examples[self.columns[0]]]
         model_inputs = self.tokenizer(inputs, max_length=1024, truncation=True)
      
         # The "labels" are the tokenized outputs:
         labels = self.tokenizer(text_target=examples[self.columns[1]], 
                           max_length=1024,         
                           truncation=True)

         model_inputs["labels"] = labels["input_ids"]
         return model_inputs
   def compute_metrics(self,eval_preds):
         preds, labels = eval_preds

         # decode preds and labels
         labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
         decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
         decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

         # rougeLSum expects newline after each sentence
         decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
         decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

         result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
         file_path=f"results_txt/{self.MODEL_NAME.replace("/","_")}"
         print("*"*10,result)
         self.results.append(result)
         if os.path.exists(file_path):
            with open(file_path, 'a') as f:
               f.write(f"{result}\n")
         else:
            with open(file_path, 'w') as f:
               f.write(f"{result}\n")
         return result
   def finetune(self,epoch,batch_size,add_prefix):
      
      # data_location = r"data/finetune_data.xlsx" 
      # data_df=pd.read_excel(self.data_location)
      dataset = Dataset.from_pandas(self.data_df)
      # We prefix our tasks with "answer the question"
      self.prefix = add_prefix #"Please answer this question: "

      # Define the preprocessing function
      dataset = dataset.train_test_split(test_size=0.1)
      # Map the preprocessing function across our dataset
      tokenized_dataset = dataset.map(self.preprocess_function, batched=True)

      # Global Parameters
    
      PER_DEVICE_EVAL_BATCH = 4
      WEIGHT_DECAY = 0.01
      SAVE_TOTAL_LIM = 3
      lr=5e-6
      gradient_accumulation=4
      # Set up training arguments
      training_args = Seq2SeqTrainingArguments(
         output_dir="./results",
         evaluation_strategy="epoch",
         learning_rate=lr,
         per_device_train_batch_size=batch_size,
         per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
         weight_decay=WEIGHT_DECAY,
         save_total_limit=SAVE_TOTAL_LIM,
         num_train_epochs=epoch,
         predict_with_generate=True,
         push_to_hub=False,
         gradient_accumulation_steps=gradient_accumulation
      )
      trainer = Seq2SeqTrainer(
         model=self.model,
         args=training_args,
         train_dataset=tokenized_dataset["train"],
         eval_dataset=tokenized_dataset["test"],
         tokenizer=self.tokenizer,
         data_collator=self.data_collator,
         compute_metrics=self.compute_metrics
      )
      self.training_info=trainer.train()
      current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
      full_output = f'models/{self.MODEL_NAME.split("/")[1]}_{current_time}'
      self.model.save_pretrained(full_output)
      self.model.save_pretrained(full_output)
      print("*"*10,": Model is saved!!!")
      
