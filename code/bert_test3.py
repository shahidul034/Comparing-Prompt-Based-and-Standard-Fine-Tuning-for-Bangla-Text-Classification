from datasets import Dataset
import pandas as pd
import numpy as np
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
import torch
from sklearn.metrics import f1_score
from openprompt import PromptForClassification
from transformers import AdamW
import os
import warnings
import tqdm
import logging
import wandb
os.environ['WANDB_DISABLED'] = 'true'
# wandb.init(
#     # entity ="shahidul034",
#     project=f"{dataset_name}"
# )
warnings.filterwarnings("ignore")

def result_save(file_path,string_to_write):
    if os.path.exists(file_path):
        with open(file_path, 'a') as file:
            file.write(string_to_write)
    else:
        with open(file_path, 'w') as file:
            file.write(string_to_write)
    print(string_to_write)
    

def prepare_datasets(df, class_name, n_samples, rounds=5):
    datasets = []
    for _ in range(rounds):
        df_train, df_test = [], []
        for label in class_name:
            df_temp = df[df['class_label'] == label[0]].sample(n_samples * 2, random_state=np.random.randint(10000))
            df_train.append(df_temp[:n_samples])
            df_test.append(df_temp[n_samples:n_samples*2])
        
        df_train = pd.concat(df_train, ignore_index=True)
        df_test = pd.concat(df_test, ignore_index=True)
        datasets.append((df_train, df_test))
    return datasets

def bert_model(model_root_name, model_full_name, df_train, df_test, epochs, class_name, prompt, dataset_name, n_samples,cnt): 
    dataset_train = Dataset.from_pandas(df_train)
    dataset_test = Dataset.from_pandas(df_test)

    dataset = {"train": [], "test": []}
    for cnt, (tr, te) in enumerate(zip(dataset_train, dataset_test)):
        input_example1 = InputExample(text_a=tr["text"], label=int(tr["label"]), guid=cnt)
        input_example2 = InputExample(text_a=te["text"], label=int(te["label"]), guid=cnt)
        dataset["train"].append(input_example1)
        dataset["test"].append(input_example2)

    plm, tokenizer, model_config, WrapperClass = load_plm(model_root_name, model_full_name)
    mytemplate = ManualTemplate(tokenizer=tokenizer, text=prompt)

    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                                        batch_size=4, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                        truncate_method="head")

    myverbalizer = ManualVerbalizer(tokenizer, num_classes=len(class_name), label_words=class_name)
    use_cuda = torch.cuda.is_available()
    prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
    if use_cuda:
        prompt_model = prompt_model.cuda()

    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)

    for epoch in range(epochs):
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step % (int(len(df_train) / 3)) == 1:
                print("Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)

    validation_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                             tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                                             batch_size=4, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                             truncate_method="head")
    allpreds = []
    alllabels = []
    for step, inputs in enumerate(validation_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
    std_dev = np.std(allpreds)
    f1 = f1_score(alllabels, allpreds, average='weighted')

    # file_path = f'result_new2//{dataset_name}.txt'
    # string_to_write = f"********\ncnt: {cnt}\nNumber of samples: {n_samples}\nDataset name: {dataset_name}\nEpochs: {epochs}\nmodel_root_name: {model_root_name}\nmodel_full_name: {model_full_name}\ntemplate_text: {prompt}\nAccuracy: {acc}\nstandard_deviation: {std_dev}\nf1_score: {f1}\n********\n"

    
    return acc,f1,std_dev

dataset_names_list = [
    "youtube_sentiment", #0
    "SentNoB", #1
    "multichannel_bsentiment", #2
    "BanglaBook", #3
    "ABSA_Datasets", #4
    "BSACD", #5
    "dsfsa", #6
    "5_Sentiment_Dataset", #7
    "CogniSenti" #8
]
dataset_num=8
data = pd.read_excel(f"testing_data2/{dataset_names_list[dataset_num]}.xlsx")

data = data.dropna()
data = data[~data['text'].apply(lambda x: isinstance(x, int))]

dataset_name = f"{dataset_names_list[dataset_num]}"+"_bert"
class_labels = [[str(x)] for x in set(data['class_label'])]

model_names = [
    ["roberta", "xlm-roberta-large"],
    ["bert", "google-bert/bert-base-multilingual-cased"],
    ["bert", "distilbert/distilbert-base-multilingual-cased"],
    ["roberta", "xlm-roberta-base"],
    ["bert", "csebuetnlp/banglabert"]
]
prompts = [
    '{"placeholder":"text_a"}. Sentiment in Bangla: {"mask"}.',
    '{"placeholder":"text_a"}. এই বাক্যটির অনুভূতি হলো {"mask"}.',
    'নিম্নলিখিত পাঠ্যের অনুভূতি হল {"mask"}: {"placeholder":"text_a"}'
]
chunk_sizes = [4,8, 16,32]
epochs=10
runtime_count=1
import time
flag=0
start_time = time.perf_counter()
for idx1,(model_root, model_full) in enumerate(model_names):
    for idx2,n_samples in enumerate(chunk_sizes):
        for idx2,prompt_text in enumerate(prompts):
            datasets = prepare_datasets(data, class_labels, n_samples)
            cnt=1
            t_acc,t_f1,t_std=0,0,0
            for idx3,(df_train, df_test) in enumerate(datasets):
                acc,f1,std=bert_model(model_root, model_full, df_train, df_test, epochs, class_labels, prompt_text, dataset_name, n_samples,cnt)
                t_acc,t_f1,t_std,cnt=t_acc+acc,t_f1+f1,t_std+std,cnt+1
            string_to_write = f"********Chunk_no: {runtime_count}\nNumber of samples: {n_samples}\nDataset name: {dataset_name}\nEpochs: {epochs}\nmodel_root_name: {model_root}\nmodel_full_name: {model_full}\ntemplate_text: {prompt_text}\nAccuracy: {t_acc/5.0}\nstandard_deviation: {t_std/5.0}\nf1_score: {t_f1/5.0}\n********\n"
            print("Chunk completed: ","*"*15,":",runtime_count,f"/{len(model_names)*len(prompts)*4}")
            runtime_count+=1

            file_path = f'modified_prompt//{dataset_name}.txt'
            
            result_save(file_path,string_to_write)
            print(string_to_write)

end_time = time.perf_counter()
total_run_time = end_time - start_time
total_run_hours = int(total_run_time // 3600)
total_run_minutes = int((total_run_time % 3600) // 60)
total_run_seconds = total_run_time % 60
print(f"Total program run time: {total_run_hours} hours, {total_run_minutes} minutes, {total_run_seconds:.2f} seconds")