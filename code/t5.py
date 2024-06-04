from datasets import load_from_disk, Dataset,concatenate_datasets
import pandas as pd
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.plms import T5TokenizerWrapper
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
import torch
from openprompt import PromptForClassification
from transformers import  AdamW, get_linear_schedule_with_warmup
import os

def t5_model(model_full_name,n_samples,epochs):

    dataset_path=r"data/Train.csv"
    model_root_name="t5"
    # model_full_name="google/mt5-base"
    # n_samples = 8

    df=pd.read_csv(dataset_path)


    df_neutral = df[df['Label'] == 0].sample(n=n_samples, random_state=1)
    df_positive = df[df['Label'] == 1].sample(n=n_samples, random_state=1)
    df_negative = df[df['Label'] == 2].sample(n=n_samples, random_state=1)


    df_sampled = pd.concat([df_neutral, df_positive, df_negative], ignore_index=True)


    dataset = Dataset.from_pandas(df_sampled)
    raw_dataset=dataset.train_test_split(test_size=0.5)

    dataset = {}
    for split in ['train', 'test']:
        dataset[split] = []
        for idx,data in enumerate(raw_dataset[split]):
            input_example = InputExample(text_a = data['Data'], label=int(data['Label']), guid=idx)
            dataset[split].append(input_example)
    plm, tokenizer, model_config, WrapperClass = load_plm(model_root_name,model_full_name )

    # template_text = '{"placeholder":"text_a"}. এটা হল {"mask"}.'

    template_text = '{"placeholder":"text_a"}. sentiment: {"mask"}.'
    # template_text = '[CLS] {"placeholder":"text_a"}. Sentiment: {"mask"}. [SEP]'
    mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<mask>']})
    wrapped_t5tokenizer= T5TokenizerWrapper(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")


    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
        batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="head")



    # for example the verbalizer contains multiple label words in each class
    myverbalizer = ManualVerbalizer(tokenizer, num_classes=3,
                            label_words=[["Neutral"], ["Positive"], ["Negative"]])



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
            if step %100 ==1:
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
    file_path = 'result.txt'
    string_to_write = f"********\nNumber of samples per class: {n_samples}\nEpochs: {epochs}\nmodel_root_name: {model_root_name}\nmodel_full_name: {model_full_name}\ntemplate_text: {template_text}\nAccuracy: {acc}\n********\n"

    if os.path.exists(file_path):
        with open(file_path, 'a') as file:
            file.write(string_to_write)
    else:
        with open(file_path, 'w') as file:
            file.write(string_to_write)
    print(f"acc: {acc}")