import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn_structure import GNN

from tqdm import tqdm
import argparse
import time
import numpy as np
import sys
import random
### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train(model, device, loader, optimizer, task_type):
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
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss_accum += loss.item()
            loss.backward()
            optimizer.step()    
    train_loss = loss_accum/(step+1)
    print('Train Loss',train_loss)
    return train_loss
        

def eval(model, device, loader, evaluator,task_type):
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
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss_accum += loss.item()

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict),loss_accum/(step+1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=7,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--lr', type=int, default=0.001,
                        help='learning rate')
    parser.add_argument('--gnn', type=str, default='gin',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=4,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=400,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--training_ratio',type=float,default=1,
                        help="between 0 and 1")
    parser.add_argument('--feature', type=str, default="simple",
                        help='full feature or simple feature')
    parser.add_argument('--random_seed',type=int,default=7)
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = "ogbg-molhiv")

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)
    #perm_index = torch.randperm(len(dataset))
    # train_index = perm_index[:int(args.training_ratio*len(dataset))]
    # val_index = perm_index[int(0.8*len(dataset)):int(0.9*len(dataset))]
    # test_index = perm_index[int(0.9*len(dataset)):]
    #idx = [i for i in range(len(dataset))]
    #split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    #evaluator = accuracy

    # train_loader = DataLoader(dataset[train_index], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    # valid_loader = DataLoader(dataset[val_index], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    # test_loader = DataLoader(dataset[test_index], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    train_index = split_idx["train"]
    print(len(train_index))
    train_loader = DataLoader(dataset[train_index[:int(args.training_ratio*len(train_index))]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
   
    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')
    
    import wandb
    wandb.init(
    # set the wandb project where this run will be logged
    project="molhiv_structure_only",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "architecture": args.gnn,
    "dataset": args.dataset,
    "epochs": args.epochs,
    "drop_ratio": args.drop_ratio,
    "num_layer": args.num_layer,
    "emb_dim": args.emb_dim,
    "batch_size": args.batch_size,
    "training_ratio": args.training_ratio,
    "random": args.random_seed
    }
    )
    valid_curve = []
    test_curve = []
    #train_curve = []
    val_loss_curve = []
    test_loss_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_loss=train(model, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        #train_perf = eval(model, device, train_loader, evaluator)
        valid_perf,val_loss = eval(model, device, valid_loader, evaluator, dataset.task_type)
        test_perf, test_loss = eval(model, device, test_loader, evaluator, dataset.task_type)

        print({'Train loss': train_loss, 'Validation': valid_perf, 'Test': test_perf})

        #train_curve.append(train_loss[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])
        val_loss_curve.append(val_loss)
        test_loss_curve.append(test_loss)
        wandb.log({"train loss":train_loss,"val acc": valid_perf, "val loss": val_loss, "test acc": test_perf, "test loss":test_loss})


    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        #best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        #best_train = min(train_curve)
    best_val_loss_epoch = np.argmin(np.array(val_loss_curve))
    

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))
    print('Best val loss: {}'.format(val_loss_curve[best_val_loss_epoch]))
    print('Best test loss: {}'.format(test_loss_curve[best_val_loss_epoch]))
    wandb.log({"para num": num_params, "best val score": valid_curve[best_val_epoch], "best test score":test_curve[best_val_epoch],
               "best val loss":val_loss_curve[best_val_loss_epoch],"best test loss":test_loss_curve[best_val_loss_epoch]})


if __name__ == "__main__":
    main()
