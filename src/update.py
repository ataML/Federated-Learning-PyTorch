#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import  Dataset
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import DataLoader
from monai.transforms import Compose,Activations,AsDiscrete
from monai.apps import DecathlonDataset

import numpy as np 
import tqdm


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        #super().__init__()
    
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        
        #print(f"itme = {item},{self.idxs},indx[item]= {self.idxs[item]},")
       # print(self.dataset[self.idxs[item]])
      #  print(type(self.dataset[self.idxs[item]]))

###        image = self.dataset[self.idxs[item]]["image"]
###        label = self.dataset[self.idxs[item]]["label"]
       # return torch.tensor(image), torch.tensor(label)
        image = self.dataset[item]["image"]
        label = self.dataset[item]["label"]
        
        return image,label


class LocalUpdate(object):
    
    def __init__(self, train_dataset,val_dataset, train_idxs,val_idxs, logger, local_bs,lr,local_ep,total_batches):
        #self.args = args
        self.logger = logger
        self.device = 'cuda' #if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion =  DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)#nn.NLLLoss().to(self.device)
        self.lr = lr
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.total_batches= total_batches
        self.trainloader, self.validloader = self.train_val_test(train_dataset,val_dataset,
                                                              list(train_idxs),list(val_idxs))
  ###      self.trainloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
  ###      self.validloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)
        
    def train_val_test(self, train_dataset,val_dataset, idxs_train,idxs_val):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        
        
       # idxs_train = idxs[:int(0.8*len(idxs))]
       # idxs_val = idxs[int(0.8*len(idxs)):]
        
            

###        trainloader = DataLoader(DatasetSplit(train_dataset, idxs_train),
###                                 batch_size=self.local_bs, shuffle=True,num_workers=4)
        
        trainloader = DataLoader(DatasetSplit(train_dataset, idxs_train),
                                    batch_size=self.local_bs, shuffle=True,num_workers=4)
 
    #   print(f"len trainloader:{len(trainloader)}")
        
        validloader = DataLoader(DatasetSplit(val_dataset, idxs_val),
                                 batch_size=self.local_bs,shuffle=False,num_workers=4)
    
    #    print(f"len validloader:{len(validloader)}")

        return trainloader, validloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss_values = []

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                        weight_decay=1e-5, amsgrad=True)

        for iter in range(self.local_ep):
            
            print("local epoch",iter+1)
         #   batch_loss = []
            epoch_loss = 0

            step = 0
            
            

            for batch_idx, (images, labels) in enumerate(self.trainloader):
##            for batch_data in self.trainloader:
                step += 1 
                
                if(step > self.total_batches):
                    break
                # print(f"{step}/{len(self.trainloader) }")
                images, labels = images.to(self.device), labels.to(self.device)
##                images,labels  = (
##                    batch_data["image"].to(self.device),
##                    batch_data["label"].to(self.device),
##                )
                optimizer.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

             ###   batch_loss.append(loss.item())
                epoch_loss += loss.item()
                
                print(
                    f"{step}/{len(self.trainloader)}"
                    f", train_loss: {loss.item():.4f}"
                )
                
          ##  epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)

 #           self.inference(model)
        
        
        return model.state_dict(), sum(epoch_loss_values) / len(epoch_loss_values)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        
        model.eval()
        val_loss, total, correct = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            
            
            step = 0
            
            dice_metric = DiceMetric(include_background=True, reduction="mean")
            post_trans = Compose(
                [Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
            )
            metric_sum = metric_sum_tc = metric_sum_wt = metric_sum_et = 0.0
            metric_count = (
                metric_count_tc
            ) = metric_count_wt = metric_count_et = 0
            
            
            for batch_idx, (images, labels) in enumerate(self.validloader):
##            for batch_data in self.validloader:
        
##                images, labels =  (
##                    batch_data["image"].to(self.device),
##                    batch_data["label"].to(self.device),
##                )
                step += 1
    
                images, labels = images.to(self.device), labels.to(self.device)
                
        
                # Inference
                val_outputs = model(images)
                val_outputs = post_trans(val_outputs)
                
                loss =  self.criterion(val_outputs, labels)
                val_loss += loss.item()
                
                # compute overall mean dice
                value, not_nans = dice_metric(y_pred=val_outputs, y=labels)
                not_nans = not_nans.item()
                metric_count += not_nans
                metric_sum += value.item() * not_nans
                # compute mean dice for TC
                value_tc, not_nans = dice_metric(
                    y_pred=val_outputs[:, 0:1], y=labels[:, 0:1]
                )
                not_nans = not_nans.item()
                metric_count_tc += not_nans
                metric_sum_tc += value_tc.item() * not_nans
                # compute mean dice for WT
                value_wt, not_nans = dice_metric(
                    y_pred=val_outputs[:, 1:2], y=labels[:, 1:2]
                )
                not_nans = not_nans.item()
                metric_count_wt += not_nans
                metric_sum_wt += value_wt.item() * not_nans
                # compute mean dice for ET
                value_et, not_nans = dice_metric(
                    y_pred=val_outputs[:, 2:3], y=labels[:, 2:3]
                )
                not_nans = not_nans.item()
                metric_count_et += not_nans
                metric_sum_et += value_et.item() * not_nans


          #  accuracy = correct/total
          #  return accuracy, loss
            metric = metric_sum / metric_count
          #  metric_values.append(metric)
            metric_tc = metric_sum_tc / metric_count_tc
           # metric_values_tc.append(metric_tc)
            metric_wt = metric_sum_wt / metric_count_wt
           # metric_values_wt.append(metric_wt)
            metric_et = metric_sum_et / metric_count_et
           # metric_values_et.append(metric_et)
            print(f"metric={metric}, metric_tc={metric_tc}, metric_wt={metric_wt}, metric_et={metric_et}")
            return metric, metric_tc, metric_wt, metric_et, val_loss / step


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """
    
    with torch.no_grad():
        
        

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        step=0

        device = 'cuda' #if args.gpu else 'cpu'
        #criterion = nn.NLLLoss().to(device)
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        post_trans = Compose(
                [Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
        )
        metric_sum = metric_sum_tc = metric_sum_wt = metric_sum_et = 0.0
        metric_count = (
                metric_count_tc
        ) = metric_count_wt = metric_count_et = 0
            
        testloader = DataLoader(test_dataset, batch_size=2,
                            shuffle=False)

        for batch_idx, (images, labels) in enumerate(testloader):
        
            images, labels = images.to(device), labels.to(device)

            step += 1
            val_outputs = model(images)
            val_outputs = post_trans(val_outputs)
              
                
            loss_val = self.criterion(val_outputs, labels)
            loss += loss_val.item()
    
            # compute overall mean dice
            value, not_nans = dice_metric(y_pred=val_outputs, y=labels)
            not_nans = not_nans.item()
            metric_count += not_nans
            metric_sum += value.item() * not_nans
            # compute mean dice for TC
            value_tc, not_nans = dice_metric(
                y_pred=val_outputs[:, 0:1], y=labels[:, 0:1]
            )
            not_nans = not_nans.item()
            metric_count_tc += not_nans
            metric_sum_tc += value_tc.item() * not_nans
            # compute mean dice for WT
            value_wt, not_nans = dice_metric(
                y_pred=val_outputs[:, 1:2], y=labels[:, 1:2]
            )
            not_nans = not_nans.item()
            metric_count_wt += not_nans
            metric_sum_wt += value_wt.item() * not_nans
            # compute mean dice for ET
            value_et, not_nans = dice_metric(
                y_pred=val_outputs[:, 2:3], y=labels[:, 2:3]
            )
            not_nans = not_nans.item()
            metric_count_et += not_nans
            metric_sum_et += value_et.item() * not_nans

        
        loss /= step
        
        metric = metric_sum / metric_count
          #  metric_values.append(metric)
        metric_tc = metric_sum_tc / metric_count_tc
           # metric_values_tc.append(metric_tc)
        metric_wt = metric_sum_wt / metric_count_wt
           # metric_values_wt.append(metric_wt)
        metric_et = metric_sum_et / metric_count_et
           # metric_values_et.append(metric_et)
            
        return metric, metric_tc, metric_wt, metric_et,loss
