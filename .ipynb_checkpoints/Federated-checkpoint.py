# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import tempfile
import copy 
import src.update 

import tqdm

import matplotlib.pyplot as plt
import numpy as np
from monai.apps import DecathlonDataset
from monai.config import print_config

from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet

from src.update import LocalUpdate
from monai.utils import set_determinism
from src.utils import get_dataset,average_weights

import torch

print_config()


#from tensorboardX import SummaryWriter

path_project = os.path.abspath('..')
#logger = SummaryWriter('../logs')
logger=0
os.environ["MONAI_DATA_DIRECTORY"]="/data"  
root_dir = os.environ.get("MONAI_DATA_DIRECTORY")

#args = args_parser()
epochs = 200
num_users = 1
frac = 1 
local_ep = 1
local_bs = 2
lr = 1e-4
momentum = 0.5
iid=1
dataset="brats"
#exp_details(args)

device = torch.device("cuda:0")

train_dataset, val_dataset, user_groups_train, user_groups_val = get_dataset(iid, num_users, download_dataset=False)



#Build model
global_model = UNet(
    dimensions=3,
    in_channels=4,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

import importlib
import src.update
importlib.reload(src.update)
from src.update import LocalUpdate
import src.utils
importlib.reload(src.utils)
from src.utils import average_weights
import time 


start = time.time()


##### global_model.to(device)
global_model.train()

#print(global_model)

# copy weights
global_weights = global_model.state_dict()

# Training
train_loss, train_accuracy,val_loss_list = [], [], []
val_acc_list, net_list = [], []
cv_loss, cv_acc = [], []
print_every = 1
val_loss_pre, counter = 0, 0


#####
best_metric = -1
best_metric_epoch = -1
#epoch_loss_values = []
metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []

metric, metric_tc, metric_wt, metric_et=0.0,0.0,0.0,0.0



total_batches = 10000




for epoch in tqdm.tqdm(range(epochs)):
        
    local_weights, local_losses = [], []
    
    print(f'\n | Global Training Round : {epoch+1} |\n')

    global_model.train()
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)

    for idx in idxs_users:
        
        print(f"user {idx} selected")
        local_model = LocalUpdate(train_dataset=train_dataset, val_dataset=val_dataset,
                                  train_idxs=user_groups_train[idx],val_idxs=user_groups_val[idx], logger=logger, local_bs=local_bs, lr=lr, local_ep=local_ep, total_batches=total_batches)
        w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
        local_weights.append(copy.deepcopy(w))
      #  local_losses.append(copy.deepcopy(loss))
        local_losses.append(loss)
     #   global_model.load_state_dict(copy.deepcopy(w))
    
     #   global_model.eval()
        
     #   local_model.inference(model=global_model)

    # update global weights
    global_weights = average_weights(local_weights)
   ## global_weights = local_weights[0]
    
    # update global loss
    global_model.load_state_dict(global_weights)
    
 #   for c in range(num_users):
 #       local_model = LocalUpdate(train_dataset=train_dataset, val_dataset=val_dataset,
  #                                  train_idxs=user_groups_train[c],val_idxs=user_groups_val[c], logger=logger, lr=lr, local_ep=local_ep, local_bs=local_bs)
   #     local_model.inference(model=global_model)
    
    loss_avg = sum(local_losses) / len(local_losses)
    train_loss.append(loss_avg)
    print(f"current epoch: {epoch + 1} current epoch loss: {loss_avg:.4f}")
    # Calculate avg training accuracy over all users at every epoch
    
   # list_acc, list_loss = [], []
    ep_metric_values = []
    ep_metric_values_tc = []
    ep_metric_values_wt = []
    ep_metric_values_et = []
    
    ep_loss_values = []

    for c in range(num_users):
        local_model = LocalUpdate(train_dataset=train_dataset, val_dataset=val_dataset,
                                    train_idxs=user_groups_train[c],val_idxs=user_groups_val[c], logger=logger, lr=lr, local_ep=local_ep, local_bs=local_bs,total_batches=total_batches)
        metric, metric_tc, metric_wt, metric_et,val_loss=local_model.inference(model=global_model)
        
        ep_metric_values.append(metric)
        ep_metric_values_tc.append(metric_tc)
        ep_metric_values_wt.append(metric_wt) 
        ep_metric_values_et.append(metric_et) 
        ep_loss_values.append(val_loss)
        
    metric = sum(ep_metric_values)/len(ep_metric_values)
    metric_tc = sum(ep_metric_values_tc)/len(ep_metric_values_tc)
    metric_wt = sum(ep_metric_values_wt)/len(ep_metric_values_wt)
    metric_et = sum(ep_metric_values_et)/len(ep_metric_values_et)
    val_loss = sum(ep_loss_values)/len(ep_loss_values)
    
    metric_values.append(metric)
    metric_values_tc.append(metric_tc)
    metric_values_wt.append(metric_wt)
    metric_values_et.append(metric_et)
    val_loss_list.append(val_loss)
    global_model.eval()
  ###  for c in range(num_users):
  ###      local_model = LocalUpdate(train_dataset=train_dataset, val_dataset=val_dataset,
  ###                                    train_idxs=user_groups_train[c],val_idxs=user_groups_val[c], logger=logger, lr=lr, local_ep=local_ep, local_bs=local_bs)
  ###      metric, metric_tc, metric_wt, metric_et = local_model.inference(model=global_model)
         
   ##    print(metric)
   ##    ep_metric_values.append(metric)
   ##     ep_metric_values_tc.append(metric_tc)
   ##     ep_metric_values_wt.append(metric_wt) 
   ##     ep_metric_values_et.append(metric_et) 

        
   # train_accuracy.append(sum(list_acc)/len(list_acc))
##    metric = sum(ep_metric_values)/len(ep_metric_values)
    print(metric)
##    metric_tc = sum(ep_metric_values_tc)/len(ep_metric_values_tc)
##    metric_wt = sum(ep_metric_values_wt)/len(ep_metric_values_wt)
##    metric_et = sum(ep_metric_values_et)/len(ep_metric_values_et)
##    metric_values.append(metric)
##    metric_values_tc.append(metric_tc)
##    metric_values_wt.append(metric_wt)
##    metric_values_et.append(metric_et)
    
    if metric > best_metric:
         
        best_metric = metric
        best_metric_epoch = epoch + 1
        torch.save(
                global_model.state_dict(),
                os.path.join(root_dir, "best_metric_model.pth"),
        )
        
     #   torch.save({
      #      'epoch': epoch + 1,
       #     'model_state_dict': model.state_dict(),
        #    'optimizer_state_dict':  optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
         #                               weight_decay=1e-5, amsgrad=True),
          #  'loss': loss,
          #  ...
           # }, root_dir)
        
        print("saved new best metric model")


    # print global training loss after every 'i' rounds
    if (epoch+1) % print_every == 0:
      #  print(f' \nAvg Training Stats after {epoch+1} global rounds:')
      #  print(f'Training Loss : {np.mean(np.array(train_loss))}')
      #  print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
         print(
            f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
            f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
            f"\nbest mean dice: {best_metric:.4f}"
            f" at epoch: {best_metric_epoch}"
         )

end = time.time() 
print(end - start)


import pickle 
with open("train_loss.pkl", 'wb') as f:
      pickle.dump(train_loss,f)
with open("metric_values.pkl", 'wb') as f:
      pickle.dump(metric_values,f)
with open("metric_values_tc.pkl", 'wb') as f:
      pickle.dump(metric_values_tc,f)
with open("metric_values_wt.pkl", 'wb') as f:
      pickle.dump(metric_values_wt,f)
with open("metric_values_et.pkl", 'wb') as f:
      pickle.dump(metric_values_et,f)
with open("val_loss.pkl",'wb') as f:
      pickle.dump(metric_values_et,f)
    
