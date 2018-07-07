# -*- coding: utf-8 -*-
"""
Created on Tuesday Jun 25 13:34:42 2018

@author: lux32
"""

from data import load_data
import os
import time
import torch
import model as net
from torch.autograd import Variable
from Criteria import CrossEntropyLoss2d
import torch.backends.cudnn as cudnn
from graphviz import Digraph
from argparse import ArgumentParser
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import DataSet as myDataLoader


def val(args, val_loader, model, criterion):

    model.eval()

    epoch_loss = []

    total_batches = len(val_loader)
    for i, (input, target) in enumerate(val_loader):
        start_time = time.time()

        if args.onGPU == True:
            input = input.cuda()
            target = target.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # run the mdoel
        output = model(input_var)

        # compute the loss
        loss = criterion(output, target_var)

        epoch_loss.append(loss.data[0])

        time_taken = time.time() - start_time

        print('[%d/%d] loss: %.3f time: %.2f' % (i, total_batches, loss.data[0], time_taken))

        average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
        mean_squared_error_score = mean_squared_error(target_var, output)

    return average_epoch_loss_val, mean_squared_error_score

def train(args, train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()

    epoch_loss = []

    total_batches = len(train_loader)
    print('------------------------------------------')
    print(enumerate(train_loader))
    for i, (input, target) in enumerate(train_loader):
        start_time = time.time()

        if args.onGPU == True:
            input = input.cuda()
            target = target.cuda()

        print(i)
        print(input.shape)
        print(len(target))
        print(len(train_loader))
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # run the mdoel
        output = model(input_var)

        optimizer.zero_grad()
        # compute the loss
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.data[0])

        time_taken = time.time() - start_time

        print('[%d/%d] loss: %.3f time: %.2f' % (i, total_batches, loss.data[0], time_taken))

        average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
        mean_squared_error_score = mean_squared_error(target_var, output)

    return average_epoch_loss_val, mean_squared_error_score

def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    torch.save(state, filenameCheckpoint)

def make_dot(var):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, Variable):
                value = '(' + (', ').join(['%d' % v for v in var.size()]) + ')'
                dot.node(str(id(var)), str(value), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'previous_functions'):
                for u in var.previous_functions:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])
        add_nodes(var.creator)
    return dot

def trainRegression(args):

    sequ, label = load_data(args.data_dir)

    train_sequ, test_sequ,  train_label, test_label = train_test_split(sequ, label, test_size=0.33, random_state=42)
    train_sequ, val_sequ,  train_label, val_label= train_test_split(train_sequ, train_label, test_size=0.33, random_state=42)

    print(train_sequ[0].shape)

    train_data_load = torch.utils.data.DataLoader(myDataLoader.MyDataset(train_sequ, train_label),
                                                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_data_load = torch.utils.data.DataLoader(myDataLoader.MyDataset(val_sequ, val_label),
                                                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    test_data_load = torch.utils.data.DataLoader(myDataLoader.MyDataset(test_sequ, test_label),
                                                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


    print("DataSet prepared")

    args.savedir = args.savedir + os.sep

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    model = net.shallow_net()

    if args.onGPU == True:
        model = model.cuda()

    total_paramters = 0
    if args.visualizeNet == True:
        x = Variable(torch.randn(1, 20, 30, 1))

        if args.onGPU == True:
            x = x.cuda()

        y = model.forward(x)
        g = make_dot(y)
        g.render(args.savedir + '/model.png', view=False)

        total_paramters = 0
        for parameter in model.parameters():
            i = len(parameter.size())
            p = 1
            for j in range(i):
                p *= parameter.size(j)
            total_paramters += p

        print('Parameters: ' + str(total_paramters))

    criteria = CrossEntropyLoss2d()

    if args.onGPU == True:
        criteria = criteria.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if args.onGPU == True:
        cudnn.benchmark = True

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t%s\t%s\t%s\t%s\t" % ('Epoch', 'Loss(Tr)', 'Loss(val)', 'MSE (tr)', 'MSE (val)'))
        logger.flush()
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t%s\t%s\t%s\t%s\t" % ('Epoch', 'Loss(Tr)', 'Loss(val)', 'MSE (tr)', 'MSE (val)'))
        logger.flush()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_loss, gamma=0.1)  #.ReduceLROnPlateau(optimizer, 'min', patience=5)

    start_epoch = 0
    best_MSE = 10000

    for epoch in range(start_epoch, args.max_epochs):
        scheduler.step(epoch)

        lr = 0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        tr_epoch_loss, tr_mean_squared_error = train(args, train_data_load, model, criteria, optimizer, epoch)
        val_epoch_loss, val_mean_squared_error = val(args, val_data_load, model, criteria, optimizer, epoch)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lossTr': tr_epoch_loss,
            'lossVal': val_epoch_loss,
            'MSETr': tr_mean_squared_error,
            'MSEVal':val_mean_squared_error,
        }, args.savedir + '/checkpoint.pth.tar')

        # save the best model
        if val_mean_squared_error < best_MSE:
            model_file_name = args.savedir + '/best_peptide_model.pth'
            print('==> Saving the best model')
            torch.save(model.state_dict(), model_file_name)
            best_MSE = val_mean_squared_error

        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f"
                     % (epoch, tr_epoch_loss, val_epoch_loss, tr_mean_squared_error, val_mean_squared_error, lr))
        logger.flush()
        print("Epoch : " + str(epoch) + ' Details')
        print("\nEpoch No.: %d\tTrain Loss = %.4f\tVal Loss = %.4f\t MSE(tr) = %.4f\t MSE(val) = %.4f" % (
                epoch, tr_epoch_loss, val_epoch_loss, tr_mean_squared_error, val_mean_squared_error))

        logger.close()





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="shallow_net")  #
    parser.add_argument('--data_dir', default="./data/")  # data directory
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=15)
    parser.add_argument('--step_loss', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--savedir', default='./results_enc_')
    parser.add_argument('--visualizeNet', type=bool, default=False)
    #parser.add_argument('--resume', type=bool,
                         #   default=False)  # Use this flag to load the last checkpoint for training
    #parser.add_argument('--resumeLoc', default='./results_enc_C1/checkpoint.pth.tar')
    parser.add_argument('--logFile', default='trainValLog.txt')
    parser.add_argument('--onGPU', default=False)

    trainRegression(parser.parse_args())


