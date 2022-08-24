import torch
import torch.nn as nn


def loss_fn(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)


def train_fn(data_loader, model, optimizer, device, scheduler, heads):
    model.train()
    finds = {}
    
    for i, batch in enumerate(data_loader):
        j = 0
        
        optimizer.zero_grad()
        
        while j < len(heads):
            head = heads.split('-')[j]
            j =+ 1
            
            if i < 1:
                finds[head]['targets'] = []
                finds[head]['predictions'] = []
                finds[head]['loss'] = 0
            
            batch = {k:v.to(device, dtype=torch.long) for k,v in batch.items()}
            targets = batch["targets"]
            del batch["targets"]

            outputs = model(batch, head)
            loss = loss_fn(outputs, targets)
            finds[head]['loss'] += loss.cpu().detach().numpy().tolist()/len(batch)
            
            finds[head]['targets'].extend(targets.cpu().detach().numpy().tolist())
            _, predictions = torch.max(outputs, 1)
            finds[head]['predictions'].extend(predictions.cpu().detach().numpy().tolist())
            
            loss.backward()
        optimizer.step()
        scheduler.step()

    return finds


def eval_fn(data_loader, model, device, heads):
    model.eval()
    finds = {}
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            j = 0
            while j < len(heads):
                head = heads.split('-')[j]
                j =+ 1
                
                if i <= 1:
                    finds[head]['targets'] = []
                    finds[head]['predictions'] = []
                    finds[head]['loss'] = 0
                
                batch = {k:v.to(device, dtype=torch.long) for k,v in batch.items()}
                targets = batch["targets"]
                del batch["targets"]

                outputs = model(batch, head)
                loss = loss_fn(outputs, targets)
                finds[head]['loss'] += loss.cpu().detach().numpy().tolist()/len(batch)
                
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