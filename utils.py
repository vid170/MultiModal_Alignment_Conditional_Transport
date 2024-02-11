from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import ViTModel
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report,f1_score,accuracy_score,precision_score, recall_score, multilabel_confusion_matrix
import numpy as np
import torch

def load_pretrained_models(hidden_dim=768):
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224', output_hidden_states=True)
    bert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",output_hidden_states=True)
    return vit_model, bert_tokenizer,bert_model

def get_data_loaders(train_ds,val_ds,batch_size=16):
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    return train_loader,val_loader

def get_metrics(predicted_classes,labels,epoch):
    pred=predicted_classes.cpu().detach().numpy()
    true=labels.cpu().detach().numpy()
    if(epoch==9):
        print(classification_report(pred,true))
    avg_recall=recall_score(pred,true,average='macro')
    avg_precision=precision_score(pred,true,average='macro')
    avg_f1=f1_score(pred,true,average='macro')    
    overall_recall=recall_score(pred,true,average='micro')
    overall_precision=precision_score(pred,true,average='micro')
    overall_f1=f1_score(pred,true,average='micro')
    return avg_recall,avg_precision,avg_f1,overall_recall,overall_precision,overall_f1  