import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from mol_gnn import Mol_GNN
from ppa_gnn import PPA_GNN
from tqdm import tqdm
import argparse
import time
import numpy as np
import sys
import random
### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()
multicls_criterion = torch.nn.CrossEntropyLoss()

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def train(model, device, loader, optimizer, task_type, dataset):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if(step%100==0):
            print("step",step)
            sys.stdout.flush()
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            #print(batch.y)
            if "classification" in task_type: 
                if dataset != "ogbg-ppa":
                    loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                else:
                    loss = multicls_criterion(pred.to(torch.float32), batch.y.view(-1,)) 
            else:
                loss = reg_criterion(pred.view(batch.y.shape).to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])

            loss_accum += loss.item()
            loss.backward()
            optimizer.step()    
    train_loss = loss_accum/(step+1)
    print('Train Loss', train_loss)
    return train_loss
        

def eval(model, device, loader, evaluator,task_type, dataset):
    model.eval()
    y_true = []
    y_pred = []
    loss_accum = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                if dataset != "ogbg-ppa":
                    loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                else:
                    loss = multicls_criterion(pred.to(torch.float32), batch.y.view(-1,))
 
            else:
                 loss = reg_criterion(pred.view(batch.y.shape).to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])

            loss_accum += loss.item()

            if dataset =='ogbg-ppa':
                y_true.append(batch.y.view(-1,1).detach().cpu())
                y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())
            else:
                y_true.append(batch.y.detach().cpu())
                y_pred.append(pred.view(batch.y.shape).detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict), loss_accum/(step+1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Test the scaling behaviors of GNNs')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--lr', type=int, default=0.001,
                        help='learning rate')
    parser.add_argument('--gnn', type=str, default='gin',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=100,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="pcqv2",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--training_ratio',type=float,default=0.01,
                        help="between 0 and 1")
    parser.add_argument('--random_seed',type=int,default=7)
    parser.add_argument('--scale_type', type=str, default="data",
                        help='data or model')
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device("cpu")
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    ### automatic dataloading and splitting
    if args.dataset == "ogbg-ppa":
        dataset = PygGraphPropPredDataset(name = args.dataset, transform=add_zeros)
    if args.dataset == "pcqv2":    
        dataset = PygPCQM4Mv2Dataset(root = 'dataset/')
    else:
        dataset = PygGraphPropPredDataset(name = args.dataset)


    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    if args.dataset != "pcqv2":
        evaluator = Evaluator(args.dataset)
    else:
        evaluator = PCQM4Mv2Evaluator()
        evaluator.eval_metric ='mae'

    train_index = split_idx["train"]
    print('The total number of graphs:', len(train_index))
    train_loader = DataLoader(dataset[train_index[:int(args.training_ratio*len(train_index))]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    
    total_node_number = 0
    total_edge_number = 0
    train_set = dataset[train_index[:int(args.training_ratio*len(train_index))]]
    for d in train_set:
        total_node_number = total_node_number+int(d.x.shape[0])
        total_edge_number = total_edge_number+int(d.edge_index.shape[1])
    print('the total node number in the training set:', total_node_number)
    print('the total edge number in the training set:', total_edge_number)

    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    if args.dataset == "pcqv2":
        test_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        dataset.num_tasks = 1
        dataset.task_type = "regression"
    else:
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    

    print(dataset.num_tasks)
    if args.gnn == 'gin':
        if args.dataset != "ogbg-ppa":
            model = Mol_GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
        else:
            model = PPA_GNN(gnn_type = 'gin', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)    
    elif args.gnn == 'gcn':
        if args.dataset != "ogbg-ppa":
            model = Mol_GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
        else:
            model = PPA_GNN(gnn_type = 'gcn', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)    
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')
    
    valid_curve = []
    test_curve = []
    #train_curve = []
    val_loss_curve = []
    test_loss_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_loss=train(model, device, train_loader, optimizer, dataset.task_type, args.dataset)

        print('Evaluating...')
        #train_perf = eval(model, device, train_loader, evaluator)
        valid_perf,val_loss = eval(model, device, valid_loader, evaluator, dataset.task_type, args.dataset)
        test_perf, test_loss = eval(model, device, test_loader, evaluator, dataset.task_type,args.dataset)

        print({'Train loss': train_loss, 'Validation': valid_perf, 'Test': test_perf})

        #train_curve.append(train_loss[dataset.eval_metric])
        if args.dataset != "pcqv2":
            valid_curve.append(valid_perf[dataset.eval_metric])
            test_curve.append(test_perf[dataset.eval_metric])
        else:
            valid_curve.append(valid_perf['mae'])
            test_curve.append(test_perf['mae'])
        val_loss_curve.append(val_loss)
        test_loss_curve.append(test_loss)


    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
    best_val_loss_epoch = np.argmin(np.array(val_loss_curve))
    

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))
    print('Best val loss: {}'.format(val_loss_curve[best_val_loss_epoch]))
    print('Best test loss: {}'.format(test_loss_curve[best_val_loss_epoch]))

    #save the results for scaling curve drawing
    #save the results for scaling curve drawing
    file_path ='./results/'+args.scale_type+'_'+args.dataset+'_'+args.gnn+'_'+dataset.task_type+'_'+evaluator.eval_metric+'.npy'
    file_path = file_path.replace(" ","")
    print(file_path)
    import os
    if os.path.exists(file_path):
        print('The file exits')
        n_array = np.load(file_path)
        n_array = n_array.tolist()
    else:
        print('we create the new result file')
        n_array = [[],[]]
    if args.scale_type == 'model':
        n_array[0].append(num_params)
    else:
        n_array[0].append(total_node_number)

    n_array[1].append(test_curve[best_val_epoch])
    n_array = np.array(n_array)
    np.save(file=file_path,arr=n_array)
    print('Results Saved!')
        


if __name__ == "__main__":
    main()
