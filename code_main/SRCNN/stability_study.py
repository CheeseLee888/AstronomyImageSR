# parts of the code adapted from https://github.com/Fivefold/SRCNN/blob/main/Torch/train.py

import argparse
import os
import copy
import pandas as pd

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset

from sklearn.model_selection import KFold
from tqdm import tqdm

from models import SRCNN
# from utils import AverageMeter, calc_psnr, calc_ssim, calc_sssim
from create_dataset import AstropyDataset
import numpy as np

import sys
sys.path.append("/Users/peterli/Desktop/BS_thesis/code_main/Data Preparation")  # locally defined
from utils_custom import AverageMeter, calc_psnr, calc_ssim, calc_sssim # modify utils to utils_custom to avoid conflict with Python package 'utils'




# 本论文核心：增加损失函数设计
from weighted_loss import WeightedMSELoss




### 
#唯一需要修改的地方！！！
#还有第170行左右 损失函数的选择
###
change='10_001' # IMPORTANT: choose the change from hr to lr here, '10_0015' or '10_001' or '4_001'

save_file_name = change + '_wl_01' # 表示使用weighted loss 第1次测试

n_folds = 5 # split code into 5 folds
# k_folds = 10 # perform 10-fold cross validation on n_train_dataloader
k_folds = 3 # perform 10-fold cross validation on n_train_dataloader
# n_epochs = 15 # number of epochs for training
n_epochs = 1 # number of epochs for training

IMG_NUM=474 # how many images in the dataset
# ITER=n_epochs # number of iterations






os.makedirs("srcnn_results_"+save_file_name, exist_ok=True)


def reset_weights(nn_model):
  for l in nn_model.children():
   if hasattr(l, 'reset_parameters'):
    l.reset_parameters()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs-dir', type=str, required=False, default="outputs")
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-epochs', type=int, default=n_epochs) ################################################ KEY POINT
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = AstropyDataset(
        csv_file="srcnn_prep_images_"+change+".csv"
    )

    
    # save fold results
    nfold = KFold(n_splits=n_folds, shuffle=False)  ################################################ KEY POINT
    
    print('*******************')

    # n_ind = 0
    for n_fold, (n_train_ids, n_eval_ids) in enumerate(nfold.split(dataset)):

        # 如果该 fold 下的所有 result_all 文件都存在 → 跳过
        skip_fold = True
        for metric in ['train', 'val', 'test']:
            for measure in ['psnr', 'ssim', 'sssim']:
                path = f"srcnn_results_{save_file_name}/{metric}_results_{measure}_all_{n_fold}.csv"
                if not os.path.exists(path):
                    skip_fold = False
        if skip_fold:
            print(f"[Skip] Fold {n_fold} 已有全部输出，跳过...")
            continue


        print(f'N FOLD {n_fold}')
        print('--------------------------------')

        # savel all results
        val_results_psnr_all = np.zeros((IMG_NUM,k_folds))
        val_results_ssim_all = np.zeros((IMG_NUM,k_folds))
        val_results_sssim_all = np.zeros((IMG_NUM,k_folds))

        train_results_psnr_all = np.zeros((IMG_NUM,k_folds))
        train_results_ssim_all = np.zeros((IMG_NUM,k_folds))
        train_results_sssim_all = np.zeros((IMG_NUM,k_folds))

        test_results_psnr_all = np.zeros((IMG_NUM,k_folds))
        test_results_ssim_all = np.zeros((IMG_NUM,k_folds))
        test_results_sssim_all = np.zeros((IMG_NUM,k_folds))

        train_dataset = Subset(dataset, n_train_ids)
        eval_dataset = Subset(dataset, n_eval_ids)

        # Define data loader for training and testing data in this fold
        n_train_dataloader =  torch.utils.data.DataLoader(train_dataset,
                            batch_size=1)
        n_eval_dataloader =  torch.utils.data.DataLoader(eval_dataset,
                            batch_size=1)

        
        # save mean fold results
        train_results_psnr = {}
        train_results_ssim = {}
        train_results_sssim = {}

        val_results_psnr = {}
        val_results_ssim = {}
        val_results_sssim = {}

        test_results_psnr = {}
        test_results_ssim = {}
        test_results_sssim = {}

        # define cross validator 
        kfold = KFold(n_splits=k_folds, shuffle=False)  ################################################ KEY POINT

        print('---------------')
        k_ind = 0
        for k_fold, (k_train_ids, k_eval_ids) in enumerate(kfold.split(train_dataset)):

            print(f'K FOLD {k_fold}')
            print('--------------------------------')

            # Sample elements randomly from a given list of ids, no replacement.
            k_train_subsampler = torch.utils.data.SubsetRandomSampler(k_train_ids)
            k_eval_subsampler = torch.utils.data.SubsetRandomSampler(k_eval_ids)

            # Define data loaders for training and testing data in this fold
            k_train_dataloader = DataLoader(train_dataset, 
                            batch_size=1, sampler=k_train_subsampler,
                            num_workers=16, pin_memory=True, drop_last=True)
            k_eval_dataloader =  torch.utils.data.DataLoader(train_dataset,
                            batch_size=1, sampler=k_eval_subsampler)

            model = SRCNN().to(device)
            
            # 损失函数：均方误差
            # criterion = nn.MSELoss()
            criterion = WeightedMSELoss()

            
            # 优化器：Adam
            optimizer = optim.Adam([
                {'params': model.conv1.parameters()},
                {'params': model.conv2.parameters()},
                {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
            ], lr=args.lr)
            model.apply(reset_weights)
    
            k_best_weights = copy.deepcopy(model.state_dict())
            k_best_epoch = 0
            k_best_sssim = 0.0
            k_best_psnr = 0.0
            k_best_ssim = 0.0

            for epoch in range(args.num_epochs):
                model.train()
                epoch_losses = AverageMeter()

                with tqdm(total=(len(k_train_dataloader) - len(k_train_dataloader) % args.batch_size)) as t:
                    t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

                    for data in k_train_dataloader:
                        inputs, labels = data
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        preds = model(inputs)

                        loss = criterion(preds, labels)

                        epoch_losses.update(loss.item(), len(inputs))

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                        t.update(len(inputs))
                
                # saving model for epoch
                torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

                model.eval()
                epoch_psnr = AverageMeter()
                epoch_ssim = AverageMeter()
                epoch_sssim = AverageMeter()
                
                #data_ind = 0
                for data in k_eval_dataloader:
                    inputs, labels = data

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.no_grad():
                        preds = model(inputs).clamp(0.0, 1.0)

                    epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
                    epoch_ssim.update(calc_ssim(preds, labels), len(inputs))
                    epoch_sssim.update(calc_sssim(preds, labels), len(inputs))

                print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
                print('eval ssim: {:.2f}'.format(epoch_ssim.avg))
                print('eval sssim: {:.2f}'.format(epoch_sssim.avg))

                if epoch_sssim.avg > k_best_sssim:
                    k_best_epoch = epoch
                    k_best_sssim = epoch_sssim.avg
                    k_best_ssim = epoch_ssim.avg
                    k_best_psnr = epoch_psnr.avg
                    k_best_weights = copy.deepcopy(model.state_dict())

            print('best epoch: {}, sssim: {:.2f}'.format(k_best_epoch, k_best_sssim))
            # save_path = f'./10_001-model-fold-{n_fold}-{k_fold}.pth'
            save_path = f'./{save_file_name}-model-fold-{n_fold}-{k_fold}.pth'
            
            torch.save(k_best_weights, os.path.join(args.outputs_dir, save_path))

            # save mean validation set results
            val_results_psnr[k_fold] = k_best_psnr.detach().cpu().numpy()
            val_results_ssim[k_fold] = k_best_ssim.detach().cpu().numpy()
            val_results_sssim[k_fold] = k_best_sssim.detach().cpu().numpy()

            val_results_psnr_df = pd.DataFrame(val_results_psnr, index=[0])
            val_results_ssim_df = pd.DataFrame(val_results_ssim, index=[0])
            val_results_sssim_df = pd.DataFrame(val_results_sssim, index=[0])
     
            val_psnr_name = "srcnn_results_"+save_file_name+"/val_results_psnr_" + str(n_fold) + ".csv"
            val_ssim_name = "srcnn_results_"+save_file_name+"/val_results_ssim_" + str(n_fold) + ".csv"
            val_sssim_name = "srcnn_results_"+save_file_name+"/val_results_sssim_" + str(n_fold) + ".csv"

            # saving the dataframe
            # 这个df随着k_fold循环逐渐增加元素，直至达到k_fold个元素；因此上面存储以n_fold区分命名是合理的
            val_results_psnr_df.to_csv(val_psnr_name)
            val_results_ssim_df.to_csv(val_ssim_name)
            val_results_sssim_df.to_csv(val_sssim_name)

            # test model and obtain training results and validation results for best epoch

            model = SRCNN().to(device)

            test_psnr = AverageMeter()
            test_ssim = AverageMeter()
            test_sssim = AverageMeter()

            state_dict = model.state_dict()
            for n, p in torch.load(os.path.join(args.outputs_dir, save_path), map_location=lambda storage, loc: storage).items():
                if n in state_dict.keys():
                    state_dict[n].copy_(p)
                else:
                    raise KeyError(n)

            model.eval()

            # savel validation results
            data_ind = 0
            for data in k_eval_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    preds = model(inputs).clamp(0.0, 1.0)
                
                val_results_psnr_all[data_ind, k_ind] = calc_psnr(preds, labels)
                val_results_ssim_all[data_ind, k_ind] = calc_ssim(preds, labels)
                val_results_sssim_all[data_ind, k_ind] = calc_sssim(preds, labels)

                data_ind += 1

            # save test results

            test_psnr = AverageMeter()
            test_ssim = AverageMeter()
            test_sssim = AverageMeter()

            data_ind = 0
            for data in n_eval_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    preds = model(inputs).clamp(0.0, 1.0)

                test_results_psnr_all[data_ind, k_ind] = calc_psnr(preds, labels)
                test_results_ssim_all[data_ind, k_ind] = calc_ssim(preds, labels)
                test_results_sssim_all[data_ind, k_ind] = calc_sssim(preds, labels)

                test_psnr.update(calc_psnr(preds, labels), len(inputs))
                test_ssim.update(calc_ssim(preds, labels), len(inputs))
                test_sssim.update(calc_sssim(preds, labels), len(inputs))

                data_ind += 1

            # save train results

            train_psnr = AverageMeter()
            train_ssim = AverageMeter()
            train_sssim = AverageMeter()

            data_ind = 0
            for data in k_train_dataloader:
                inputs, labels = data 

                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    preds = model(inputs).clamp(0.0, 1.0)

                train_results_psnr_all[data_ind, k_ind] = calc_psnr(preds, labels)
                train_results_ssim_all[data_ind, k_ind] = calc_ssim(preds, labels)
                train_results_sssim_all[data_ind, k_ind] = calc_sssim(preds, labels)
            
                train_psnr.update(calc_psnr(preds, labels), len(inputs))
                train_ssim.update(calc_ssim(preds, labels), len(inputs))
                train_sssim.update(calc_sssim(preds, labels), len(inputs))
                data_ind += 1

            print('test psnr: {:.2f}'.format(test_psnr.avg))
            print('test ssim: {:.2f}'.format(test_ssim.avg))
            print('test sssim: {:.2f}'.format(test_sssim.avg))

            test_results_psnr[k_fold] = test_psnr.avg.detach().cpu().numpy()
            test_results_ssim[k_fold] = test_ssim.avg.detach().cpu().numpy()
            test_results_sssim[k_fold] = test_sssim.avg.detach().cpu().numpy()

            test_results_psnr_df = pd.DataFrame(test_results_psnr, index=[0])
            test_results_ssim_df = pd.DataFrame(test_results_ssim, index=[0])
            test_results_sssim_df = pd.DataFrame(test_results_sssim, index=[0])

            test_psnr_name = "srcnn_results_"+save_file_name+"/test_results_psnr_" + str(n_fold) + ".csv"
            test_ssim_name = "srcnn_results_"+save_file_name+"/test_results_ssim_" + str(n_fold) + ".csv"
            test_sssim_name = "srcnn_results_"+save_file_name+"/test_results_sssim_" + str(n_fold) + ".csv"

            # saving the dataframe
            test_results_psnr_df.to_csv(test_psnr_name)
            test_results_ssim_df.to_csv(test_ssim_name)
            test_results_sssim_df.to_csv(test_sssim_name)

            # save training set results
            train_results_psnr[k_fold] = train_psnr.avg.detach().cpu().numpy()
            train_results_ssim[k_fold] = train_ssim.avg.detach().cpu().numpy()
            train_results_sssim[k_fold] = train_sssim.avg.detach().cpu().numpy()

            train_results_psnr_df = pd.DataFrame(train_results_psnr, index=[0])
            train_results_ssim_df = pd.DataFrame(train_results_ssim, index=[0])
            train_results_sssim_df = pd.DataFrame(train_results_sssim, index=[0])

            train_psnr_name = "srcnn_results_"+save_file_name+"/train_results_psnr_" + str(n_fold) + ".csv"
            train_ssim_name = "srcnn_results_"+save_file_name+"/train_results_ssim_" + str(n_fold) + ".csv"
            train_sssim_name = "srcnn_results_"+save_file_name+"/train_results_sssim_" + str(n_fold) + ".csv"

            # saving the dataframe
            train_results_psnr_df.to_csv(train_psnr_name)
            train_results_ssim_df.to_csv(train_ssim_name)
            train_results_sssim_df.to_csv(train_sssim_name)

            k_ind += 1

        # save all results to csv files
        # test_psnr_all_name = "srcnn_results_"+change+"/test_results_psnr_all_" + str(n_ind) + ".csv"
        test_psnr_all_name = "srcnn_results_"+save_file_name+"/test_results_psnr_all_" + str(n_fold) + ".csv"
        test_ssim_all_name = "srcnn_results_"+save_file_name+"/test_results_ssim_all_" + str(n_fold) + ".csv"
        test_sssim_all_name = "srcnn_results_"+save_file_name+"/test_results_sssim_all_" + str(n_fold) + ".csv"

        val_psnr_all_name = "srcnn_results_"+save_file_name+"/val_results_psnr_all_" + str(n_fold) + ".csv"
        val_ssim_all_name = "srcnn_results_"+save_file_name+"/val_results_ssim_all_" + str(n_fold) + ".csv"
        val_sssim_all_name = "srcnn_results_"+save_file_name+"/val_results_sssim_all_" + str(n_fold) + ".csv"

        train_psnr_all_name = "srcnn_results_"+save_file_name+"/train_results_psnr_all_" + str(n_fold) + ".csv"
        train_ssim_all_name = "srcnn_results_"+save_file_name+"/train_results_ssim_all_" + str(n_fold) + ".csv"
        train_sssim_all_name = "srcnn_results_"+save_file_name+"/train_results_sssim_all_" + str(n_fold) + ".csv"

        train_results_psnr_all_df = pd.DataFrame(train_results_psnr_all)
        train_results_psnr_all_df.to_csv(train_psnr_all_name)
        train_results_ssim_all_df = pd.DataFrame(train_results_ssim_all)
        train_results_ssim_all_df.to_csv(train_ssim_all_name)
        train_results_sssim_all_df = pd.DataFrame(train_results_sssim_all)
        train_results_sssim_all_df.to_csv(train_sssim_all_name)

        test_results_psnr_all_df = pd.DataFrame(test_results_psnr_all)
        test_results_psnr_all_df.to_csv(test_psnr_all_name)
        test_results_ssim_all_df = pd.DataFrame(test_results_ssim_all)
        test_results_ssim_all_df.to_csv(test_ssim_all_name)
        test_results_sssim_all_df = pd.DataFrame(test_results_sssim_all)
        test_results_sssim_all_df.to_csv(test_sssim_all_name)

        val_results_psnr_all_df = pd.DataFrame(val_results_psnr_all)
        val_results_psnr_all_df.to_csv(val_psnr_all_name)
        val_results_ssim_all_df = pd.DataFrame(val_results_ssim_all)
        val_results_ssim_all_df.to_csv(val_ssim_all_name)
        val_results_sssim_all_df = pd.DataFrame(val_results_sssim_all)
        val_results_sssim_all_df.to_csv(val_sssim_all_name)

        # n_ind += 1

