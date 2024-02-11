import torch
from torch import optim
import random
import requests
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import time
import torchvision.transforms as transforms
from transformers import ViTModel,AutoTokenizer,AutoModel
from transformers import ViTImageProcessor, ViTForImageClassification
from utils import load_pretrained_models,get_data_loaders,get_metrics
from loss import CtAslLoss
from data import load_dataset
from classifier import MultiLabelClassifier
from arg_parser import parse_arguments
import warnings
import wandb

warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__=='__main__':
    args=parse_arguments()
    print("Arguments provided", args)
    torch.autograd.set_detect_anomaly(True)
    lr=1e-5
    positive_gamma=0
    negative_gamma=2
    k=20
    weight_decay=1e-2
    hidden_dim=512
    sequence_length = 197
    embedding_size = 768
    hidden_size = 256
    num_classes=48
 
    data='fitz'

    random.seed(args.seed)
    
    mode="online"
    if args.disable_wandb:
        mode="disabled"

    wandb.init(entity="multimodal12", project="Patchct", mode = mode)
    # if(args.run_name=="test_run"):
    args.run_name=f"{args.loss}_epochs{args.epochs}_datapoints{args.datapoints}_train+test_batch_size{args.batch_size}_beta{args.beta}"
    wandb.run.name = args.run_name
   

    config = wandb.config          
    config.batch_size = args.batch_size          
    config.epochs = args.epochs             
    config.opt = args.run_name
    config.datapoints=args.datapoints 


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", device)
    train_ds,val_ds=load_dataset(data,args.datapoints)
    train_loader,val_loader=get_data_loaders(train_ds,val_ds,args.batch_size)
   
    vit_model,bert_tokenizer,bert_model=load_pretrained_models()
    vit_model.to(device)
    bert_model.to(device)
    bin_classifier = MultiLabelClassifier(input_size=sequence_length*embedding_size, hidden_size=hidden_size, num_classes = num_classes)
    bin_classifier.to(device)
    loss_fn = CtAslLoss(positive_gamma,negative_gamma,k)

    vit_optimizer = optim.AdamW(vit_model.parameters(), lr=lr,weight_decay=  weight_decay)
    bert_optimizer = optim.AdamW(bert_model.parameters(), lr=lr,weight_decay=weight_decay)
    classifier_optimizer=optim.AdamW(bin_classifier.parameters(), lr=1e-3,weight_decay=weight_decay)
    class_labels=train_ds.class_labels
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    start_time=time.time()
    for epoch in range(args.epochs):
        bert_model.train()
        vit_model.train()
        bin_classifier.train()
        all_losses,all_avg_f1s,all_avg_precisions,all_avg_recalls,all_overall_f1s,all_overall_precisions,all_overall_recalls=0,0,0,0,0,0,0
        batch_count=0
    
        for batch_idx,batch in enumerate(train_loader):
            print(batch_idx, "Batch Number is =================",)
            bert_optimizer.zero_grad()
            vit_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            inputs = bert_tokenizer(class_labels, padding=True, truncation=True, return_tensors='pt')
            input_ids=inputs['input_ids'].to(device)
            attention_mask=inputs['attention_mask'].to(device)
            bert_outputs=bert_model(input_ids,attention_mask=attention_mask)
            pixel_values,labels = batch['pixel_values'].to(device),batch['label'].to(device)
            
            # inputs = processor(images=pixel_values, return_tensors="pt")
            vit_outputs = vit_model(pixel_values)
      
            logits = bin_classifier(vit_outputs.last_hidden_state)
      
            predicted_classes = (logits > 0.5).float()
            avg_recall,avg_precision,avg_f1,overall_recall,overall_precision,overall_f1=get_metrics(predicted_classes,labels,epoch)

            all_avg_recalls+=(avg_recall)
            all_avg_precisions+=(avg_precision)
            all_avg_f1s+=(avg_f1)
            all_overall_recalls   +=(overall_recall)
            all_overall_precisions+=(overall_precision)
            all_overall_f1s       +=(overall_f1)
            
            batch_count+=1
            
            if(args.loss=="ct_loss"):
                loss=loss_fn(vit_outputs,bert_outputs,labels)
            elif(args.loss=="bce_loss"):
                loss = nn.BCELoss()(logits,labels)
            elif(args.loss="steady_loss"):
                if(epoch<args.checkpoint):
                    loss=nn.BCELoss()(logits,labels)
                else:
                    loss=nn.BCE()(logits,labels)+args.beta*loss_fn(vit_outputs,bert_outputs,labels)+nn.BCELoss()(logits,labels)
            else:
                loss=args.beta*loss_fn(vit_outputs,bert_outputs,labels)+nn.BCELoss()(logits,labels)

            all_losses+=(loss)
    
            bert_old_parameter=list(bert_model.parameters())[0]
            vit_old_parameter=list(vit_model.parameters())[0]
            classifier_old_parameter=list(bin_classifier.parameters())[0]

            # bert_old,vit_old,classifier_old=[],[],[]

            # for para in list(bert_model.parameters()):
            #     bert_old.append(para.clone())
            # for para in list(vit_model.parameters()):
            #     vit_old.append(para.clone())
            # for para in list(bin_classifier.parameters()):
            #     classifier_old.append(para.clone())

            loss.backward()

            bert_optimizer.step()
            vit_optimizer.step()
            classifier_optimizer.step()
      
            # print("For BERT Model Parameters: ")
            # for i in range(len(list(bert_model.parameters()))):
            #     print(torch.equal(bert_old[i].data,list(bert_model.parameters())[i].clone().data))
            # print("For ViT Model Parameters: ")
            # for i in range(len(list(vit_model.parameters()))):
            #     print(torch.equal(vit_old[i].data,list(vit_model.parameters())[i].clone().data))
            # print("For Classifier Model Parameters: ")
            # for i in range(len(list(bin_classifier.parameters()))):
            #     print(torch.equal(classifier_old[i].data,list(bin_classifier.parameters())[i].clone().data))
          

        all_avg_f1=all_avg_f1s/batch_count
        all_avg_precision=all_avg_precisions/batch_count
        all_avg_recall=all_avg_recalls/batch_count
        all_overall_f1=all_overall_f1s/batch_count
        all_overall_precision=all_overall_precisions/batch_count
        all_overall_recall=all_overall_recalls/batch_count
        all_loss=all_losses/batch_count
        print(f"Epoch: {epoch+1}/{args.epochs}")
        print("TRAIN: Loss: {loss}, Average Recall: {avg_recall}, Average Precision: {avg_precision}, Average F1: {avg_f1}, Overall Recall: {overall_recall}, Overall Precision: {overall_precision}, Overall F1: {overall_f1}".format(avg_f1=all_avg_f1,avg_precision=all_avg_precision,avg_recall=all_avg_recall,overall_f1=all_overall_f1,overall_precision=all_overall_precision,overall_recall=all_overall_recall,loss=all_loss))
        wandb.log({
            "Epoch":epoch,
            "Train average Recall":all_avg_recall,
            "Train avergae precision": all_avg_precision,
            "Train average f1": all_avg_f1,
            "Train overall recall": all_overall_recall,
            "Train overall precision": all_overall_precision,
            "Train overall f1": all_overall_f1,
            "Train loss": all_loss,
        })

        # Validation
        bert_model.eval()
        vit_model.eval()
        bin_classifier.eval()

        with torch.no_grad():
            test_avg_f1s,test_avg_precisions,test_avg_recalls,test_overall_f1s,test_overall_precisions,test_overall_recalls=0,0,0,0,0,0
            batch_count=0
            for batch_idx,batch in enumerate(val_loader):
                pixel_values,labels = batch['pixel_values'].to(device),batch['label'].to(device)
                vit_outputs = vit_model(pixel_values)
                logits = bin_classifier(vit_outputs.last_hidden_state)
                predicted_classes = (logits > 0.5).float()
                avg_recall,avg_precision,avg_f1,overall_recall,overall_precision,overall_recall=get_metrics(predicted_classes,labels,epoch)
                test_avg_recalls+=(avg_recall)
                test_avg_precisions+=(avg_precision)
                test_avg_f1s+=(avg_f1)
                test_overall_recalls   +=(overall_recall)
                test_overall_precisions+=(overall_precision)
                test_overall_f1s       +=(overall_f1)
                batch_count+=1

            test_avg_f1=test_avg_f1s/batch_count
            test_avg_precision=test_avg_precisions/batch_count
            test_avg_recall=test_avg_recalls/batch_count
            test_overall_f1=test_overall_f1s/batch_count
            test_overall_precision=test_overall_precisions/batch_count
            test_overall_recall=test_overall_recalls/batch_count
            
        print(f"VAL: Average Recall: {test_avg_recall}, Average Precision: {test_avg_precision}, Average F1: {test_avg_f1}, Overall Recall: {test_overall_recall}, Overall Precision: {test_overall_precision}, Overall F1: {test_overall_f1}")
        
        wandb.log({
            "Epoch":epoch,
            "Test average Recall":    test_avg_recall,
            "Test avergae precision": test_avg_precision,
            "Test average f1": test_avg_f1,
            "Test overall recall": test_overall_recall,
            "Test overall precision": test_overall_precision,
            "Test overall f1": test_overall_f1,
        })

        time_elapsed=(time.time()-start_time)/60
        print("Total time taken is : ", time_elapsed)
        wandb.log({"Time taken by program in mins": time_elapsed})
