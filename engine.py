import torch
import torch.nn as nn


def loss_fn(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    finds = {}
    
    for i, batch_heads in enumerate(data_loader, start=1):
        #TODO: data loader need to retrive head plus data
        #TODO: or train_fn and val_fn need to receive heads
        for head, batch in batch_heads:
            
            if i <= 1:
                finds[head]['targets'] = []
                finds[head]['predictions'] = []
                finds[head]['loss'] = 0
            
            batch = {k:v.to(device, dtype=torch.long) for k,v in batch.items()}
            targets = batch["targets"]
            del batch["targets"]

            optimizer.zero_grad()
            outputs = model(batch, head)
            loss = loss_fn(outputs, targets)
            finds[head]['loss'] += loss.cpu().detach().numpy().tolist()/len(batch_heads)
            
            finds[head]['targets'].extend(targets.cpu().detach().numpy().tolist())
            _, predictions = torch.max(outputs, 1)
            finds[head]['predictions'].extend(predictions.cpu().detach().numpy().tolist())
            
            loss.backward()
        optimizer.step()
        scheduler.step()

    return finds


def eval_fn(data_loader, model, device):
    model.eval()
    finds = {}
    
    with torch.no_grad():
        for i, batch_heads in enumerate(data_loader, start=1):
            #TODO: data loader need to retrive head plus data
            for head, batch in batch_heads:
                
                if i <= 1:
                    finds[head]['targets'] = []
                    finds[head]['predictions'] = []
                    finds[head]['loss'] = 0
                
                batch = {k:v.to(device, dtype=torch.long) for k,v in batch.items()}
                targets = batch["targets"]
                del batch["targets"]

                outputs = model(batch, head)
                loss = loss_fn(outputs, targets)
                finds[head]['loss'] += loss.cpu().detach().numpy().tolist()/len(batch_heads)
                
                finds[head]['targets'].extend(targets.cpu().detach().numpy().tolist())
                _, predictions = torch.max(outputs, 1)
                finds[head]['predictions'].extend(predictions.cpu().detach().numpy().tolist())

    return finds
        
        
        
##########################################################################################

# mtl-2022 before adapt code fro MTL models
# def loss_fn(outputs, targets):
#     return nn.CrossEntropyLoss()(outputs, targets)


# def train_fn(data_loader, model, optimizer, device, scheduler):
#     model.train()
#     fin_targets = []
#     fin_predictions = []
#     total_loss = 0
    
    
#     for batch in data_loader:
#         batch = {k:v.to(device, dtype=torch.long) for k,v in batch.items()}
#         targets = batch["targets"]
#         del batch["targets"]
        
#         optimizer.zero_grad()
#         outputs = model(batch)
#         loss = loss_fn(outputs, targets)
#         total_loss += loss.cpu().detach().numpy().tolist()
        
#         fin_targets.extend(targets.cpu().detach().numpy().tolist())
#         _, predictions = torch.max(outputs, 1)
#         fin_predictions.extend(predictions.cpu().detach().numpy().tolist())
        
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
        
#     return fin_predictions, fin_targets, total_loss/len(data_loader)
        
        
        
# def eval_fn(data_loader, model, device):
#     model.eval()
#     fin_targets = []
#     fin_predictions = []
#     total_loss = 0
    
#     with torch.no_grad():
#         for batch in data_loader:
#             batch = {k:v.to(device, dtype=torch.long) for k,v in batch.items()}
#             targets = batch["targets"]
#             del batch["targets"]

#             outputs = model(batch)
#             loss = loss_fn(outputs, targets)
#             total_loss += loss.cpu().detach().numpy().tolist()
            
#             fin_targets.extend(targets.cpu().detach().numpy().tolist())
#             _, predictions = torch.max(outputs, 1)
#             fin_predictions.extend(predictions.cpu().detach().numpy().tolist())
    
#     return fin_predictions, fin_targets, total_loss/len(data_loader)

# CODE from OSCAT2022
# def predict_fn(data_loader, model, device):
#     model.eval()
#     fin_targets = []
#     fin_predictions = []
    
#     with torch.no_grad():
#         for batch in data_loader:
#             batch = {k:v.to(device, dtype=torch.long) for k,v in batch.items()}
#             targets = batch["targets"]
#             del batch["targets"]

#             outputs = model(batch)
            
#             fin_targets.extend(targets.cpu().detach().numpy().tolist())
#             fin_predictions.extend(outputs.cpu().detach().numpy().tolist())
    
#     return fin_predictions, fin_targets


# def test_fn(data_loader, model, device):
#     model.eval()
#     fin_predictions = []

#     with torch.no_grad():
#         for batch in data_loader:
            
#             batch = {k:v.to(device, dtype=torch.long) for k,v in batch.items()}
#             outputs = model(batch)
            
#             fin_predictions.extend(outputs.cpu().detach().numpy().tolist())
    
#     return fin_predictions