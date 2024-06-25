
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
from transformers import  AdamW
import os
import warnings
warnings.filterwarnings("ignore")
def bert_model(model_root_name,model_full_name,df,epochs,class_name,prompt,dataset_name,n_samples): 
    # dataset_path=r"data/Train.csv"
    # model_root_name="roberta"
    # model_full_name="google/mt5-base"
    # n_samples = 8

    # df=pd.read_csv(dataset_path) 
    df_train=[]
    df_test=[]
    for i in class_name: 
        df_temp = df[df['class_label'] == i[0]].sample(n_samples*2, random_state=1)
        df_train.append(df_temp[0:n_samples])
        df_test.append(df_temp[n_samples:n_samples*2])


    df_train2 = pd.concat(df_train, ignore_index=True)
    df_test2=pd.concat(df_test, ignore_index=True)

    # df = df.astype(str)
    dataset_train = Dataset.from_pandas(df_train2)
    dataset_test = Dataset.from_pandas(df_test2)
    # raw_dataset=dataset.train_test_split(test_size=0.5)

    dataset = {}
    dataset["train"]=[]
    dataset["test"]=[]
    # for split in ['train', 'test']:
    #     dataset[split] = []
    #     for idx,data in enumerate(raw_dataset[split]):
    #         input_example = InputExample(text_a = data["text"], label=int(data["label"]), guid=idx)
    #         dataset[split].append(input_example)
    cnt=0
    for tr,te in zip(dataset_train,dataset_test):
        input_example1 = InputExample(text_a = tr["text"], label=int(tr["label"]), guid=cnt)
        input_example2 = InputExample(text_a = te["text"], label=int(te["label"]), guid=cnt)
        cnt+=1

        dataset["train"].append(input_example1)
        dataset["test"].append(input_example2)
    plm, tokenizer, model_config, WrapperClass = load_plm(model_root_name,model_full_name )

    template_text = prompt #'[CLS] {"placeholder":"text_a"}. Sentiment: {"mask"}. [SEP]'
    mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
    # wrapped_MLMTokenizerWrapper= MLMTokenizerWrapper(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")


    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
        batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="head")



    # for example the verbalizer contains multiple label words in each class
    myverbalizer = ManualVerbalizer(tokenizer, num_classes=len(class_name),
                            label_words=class_name)



    use_cuda = True
    prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
    if use_cuda:
        prompt_model=  prompt_model.cuda()

    # Now the training is standard

    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
    import  tqdm
    for epoch in range(epochs):
        tot_loss = 0
        for step, inputs in tqdm.tqdm(enumerate(train_dataloader)):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step %(int(len(df)/3))==1:
                print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)

    validation_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
        batch_size=4,shuffle=False, teacher_forcing=False, predict_eos_token=False,
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

    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    print(allpreds)
    std_dev = np.std(allpreds)
    f1 = f1_score(alllabels, allpreds, average='weighted')  # Change `average` to 'micro', 'macro', etc., as needed

    file_path = f'result_new//{dataset_name}.txt'
    string_to_write = f"********\nNumber of samples: {n_samples}\nDataset name: {dataset_name}\nEpochs: {epochs}\nmodel_root_name: {model_root_name}\nmodel_full_name: {model_full_name}\ntemplate_text: {template_text}\nAccuracy: {acc}\nstandard_devidation: {std_dev}\nf1_score: {f1}\n********\n"

    if os.path.exists(file_path):
        with open(file_path, 'a') as file:
            file.write(string_to_write)
    else:
        with open(file_path, 'w') as file:
            file.write(string_to_write)
    print(f"{string_to_write}")


# prompt='[CLS] নিম্নলিখিত পাঠ্যের অনুভূতি হল {"mask"}: {"placeholder":"text_a"} [SEP]'
# prompt =  '[CLS] {"placeholder":"text_a"}. এই বাক্যটির অনুভূতি হলো {"mask"}.[SEP]'

data=pd.read_excel(r"testing_data2/BanglaBook.xlsx")
dataset_name="BanglaBook"
dd=[]
[dd.append([str(x)]) for x in set(data['class_label'])]

model_name=[["roberta","xlm-roberta-large"],["bert","google-bert/bert-base-multilingual-cased"],["bert","distilbert/distilbert-base-multilingual-cased"],
            ["roberta","xlm-roberta-base"], ["bert","csebuetnlp/banglabert"]
            ]
prompts =  ['[CLS] {"placeholder":"text_a"}. Sentiment: {"mask"}. [SEP]','[CLS] {"placeholder":"text_a"}. এই বাক্যটির অনুভূতি হলো {"mask"}.[SEP]',
'[CLS] নিম্নলিখিত পাঠ্যের অনুভূতি হল {"mask"}: {"placeholder":"text_a"} [SEP]'
        ]
chunk_size=[4,8,16,32]
bert_model(model_name[1][0], model_name[1][1], data, 5 , dd,prompts[1],"blank",4)
assert(False)
for x in model_name:
    for x2 in chunk_size:
        for x3 in prompts:
            bert_model(x[0], x[1], data, 10 , dd,x3,dataset_name,x2)










# s=set(data['tag'])
# class_map={}
# for idx,x in enumerate(s):
#     class_map[x]=idx
# data['label']=[class_map[x] for x in data['sentiment label']]
# df_shuffled2 = data.sample(frac=1).reset_index(drop=True)
# df_shuffled = df_shuffled2.sample(n=15000, random_state=42)
# print(len(df_shuffled))
# print(df_shuffled.head())
# bert_model("roberta", "xlm-roberta-large", df_shuffled, 10,["Review", "label"] , dd,prompt,dataset_name)
# bert_model("bert", "google-bert/bert-base-multilingual-cased", df_shuffled, 10,["Review", "label"], dd,prompt,dataset_name)
# bert_model("bert","distilbert/distilbert-base-multilingual-cased",df_shuffled,10,["Review", "label"],dd,prompt,dataset_name)
# bert_model("roberta", "xlm-roberta-base", df_shuffled, 10, ["Review", "label"], dd,prompt,dataset_name)
# bert_model("bert", "csebuetnlp/banglabert", df_shuffled, 10,["Review", "label"], dd,prompt,dataset_name)
# # Prompt changed
# prompt =  '[CLS] {"placeholder":"text_a"}. এই বাক্যটির অনুভূতি হলো {"mask"}.[SEP]'
# bert_model("roberta", "xlm-roberta-large", df_shuffled, 10, ["Review", "label"], dd,prompt,dataset_name)
# bert_model("bert", "google-bert/bert-base-multilingual-cased", df_shuffled, 10, ["Review", "label"], dd,prompt,dataset_name)
# bert_model("bert","distilbert/distilbert-base-multilingual-cased",df_shuffled,10,["Review", "label"],dd,prompt,dataset_name)
# bert_model("roberta", "xlm-roberta-base", df_shuffled, 10, ["Review", "label"], dd,prompt,dataset_name)
# bert_model("bert", "csebuetnlp/banglabert", df_shuffled, 10, ["Review", "label"], dd,prompt,dataset_name)

## New dataset added 2

# data=pd.read_excel(r"testing data/BSACD.xlsx")
# dataset_name="BSACD"
# dd=[]
# [dd.append([x]) for x in set(data['label'])]
# s=set(data['label'])
# class_map={}
# for idx,x in enumerate(s):
#     class_map[x]=idx
# data['label']=[class_map[x] for x in data['label']]
# df_shuffled = data.sample(frac=1).reset_index(drop=True)
# prompt =  '[CLS] {"placeholder":"text_a"}. Sentiment: {"mask"}. [SEP]'
# bert_model("roberta", "xlm-roberta-large", df_shuffled, 10, ["text", "label"], dd,prompt,dataset_name)
# bert_model("bert", "google-bert/bert-base-multilingual-cased", df_shuffled, 10, ["text", "label"], dd,prompt,dataset_name)
# bert_model("bert","distilbert/distilbert-base-multilingual-cased",df_shuffled,10,["text","label"],dd,prompt,dataset_name)
# bert_model("roberta", "xlm-roberta-base", df_shuffled, 10, ["text", "label"], dd,prompt,dataset_name)
# bert_model("bert", "csebuetnlp/banglabert", df_shuffled, 10, ["text", "label"], dd,prompt,dataset_name)

# prompt =  '[CLS] {"placeholder":"text_a"}. এই বাক্যটির অনুভূতি হলো {"mask"}.[SEP]'
# bert_model("roberta", "xlm-roberta-large", df_shuffled, 10, ["text", "label"], dd,prompt,dataset_name)
# bert_model("bert", "google-bert/bert-base-multilingual-cased", df_shuffled, 10, ["text", "label"], dd,prompt,dataset_name)
# bert_model("bert","distilbert/distilbert-base-multilingual-cased",df_shuffled,10,["text","label"],dd,prompt,dataset_name)
# bert_model("roberta", "xlm-roberta-base", df_shuffled, 10, ["text", "label"], dd,prompt,dataset_name)
# bert_model("bert", "csebuetnlp/banglabert", df_shuffled, 10, ["text", "label"], dd,prompt,dataset_name)

## New dataset added 3

# data=pd.read_excel(r"testing data/5_Sentiment_Dataset.xlsx")
# dataset_name="5_Sentiment_Dataset"
# dd=[]
# xx=[dd.append([x]) for x in set(data['Tag'])]
# s=set(data['Tag'])
# class_map={}
# for idx,x in enumerate(s):
#     class_map[x]=idx
# data['label']=[class_map[x] for x in data['Tag']]
# df_shuffled = data.sample(frac=1).reset_index(drop=True)
# print(df_shuffled.head())
# prompt =  '[CLS] {"placeholder":"text_a"}. Sentiment: {"mask"}. [SEP]'
# bert_model("roberta", "xlm-roberta-large", df_shuffled, 10, ["Text", "label"], dd,prompt,dataset_name)
# bert_model("bert", "google-bert/bert-base-multilingual-cased", df_shuffled, 10, ["Text", "label"], dd,prompt,dataset_name)
# bert_model("bert","distilbert/distilbert-base-multilingual-cased",df_shuffled,10,["Text","label"],dd,prompt,dataset_name)
# bert_model("roberta", "xlm-roberta-base", df_shuffled, 10, ["Text", "label"], dd,prompt,dataset_name)
# bert_model("bert", "csebuetnlp/banglabert", df_shuffled, 10, ["Text", "label"], dd,prompt,dataset_name)

# prompt =  '[CLS] {"placeholder":"text_a"}. এই বাক্যটির অনুভূতি হলো {"mask"}.[SEP]'
# bert_model("roberta", "xlm-roberta-large", df_shuffled, 10, ["Text", "label"], dd,prompt,dataset_name)
# bert_model("bert", "google-bert/bert-base-multilingual-cased", df_shuffled, 10, ["Text", "label"], dd,prompt,dataset_name)
# bert_model("bert","distilbert/distilbert-base-multilingual-cased",df_shuffled,10,["Text", "label"],dd,prompt,dataset_name)
# bert_model("roberta", "xlm-roberta-base", df_shuffled, 10, ["Text", "label"], dd,prompt,dataset_name)
# bert_model("bert", "csebuetnlp/banglabert", df_shuffled, 10, ["Text", "label"], dd,prompt,dataset_name)