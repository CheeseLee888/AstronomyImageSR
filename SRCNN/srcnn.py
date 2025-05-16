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

from create_dataset import AstropyDataset
import numpy as np

import sys
sys.path.append('../')  # locally defined
from utils_custom import save_output_image, AverageMeter, calc_psnr, calc_ssim, calc_sssim # modify utils to utils_custom to avoid conflict with Python package 'utils'




# add weighted loss function
from weighted_loss import CenterWeightedMAELoss
from weighted_loss import CenterWeightedMSELoss




"""
THE ONLY CELL BELOW: MODIFY VALUES TO CATER
"""

image_dataset_name = 'Hubble_Images_top90_256x256' # image dataset folder name
image_format = 'tif' # 'tif' or 'fits'
change = '4_001' #  '4_001' or '10_001' or '10_0015'

LOSS_FUNC = 'WeightedMSE' # choose loss function：'MAE' or 'MSE' or 'WeightedMAE' or 'WeightedMSE'
ALPHA = 2.0 # parameter for weighted loss function

# Below is for distinguished file name setting

# weight='mae' # MAE L1
# weight='mse' # MSE L2
weight='Wmse_02' # weighted mse, weight_map = 1.0 + 2.0 * weight_map
# weight='Wmae_04' # weighted mae, weight_map = 1.0 + 4.0 * weight_map


# set training methods
n_folds = 5 # split code into 5 folds

k_folds = 4 # perform 10-fold cross validation on n_train_dataloader

n_epochs = 10 # number of epochs for training

IMG_NUM = 90 # how many images in the dataset




save_file_name = change +  '_' + weight + '_ep' + str(n_epochs)
save_path = 'result_data_'+image_dataset_name+'/'

save_dir = save_path + save_file_name
scrnn_image_dir = '../../SRCNN_Images/'+image_dataset_name+'/'+save_file_name # output: SRCNN images

os.makedirs(save_dir, exist_ok=True)

# save settings
with open(f"{save_dir}/settings.txt", "w") as f:
    f.write(f"change = {change}\n")
    f.write(f"LOSS_FUNC = {LOSS_FUNC}\n")
    if LOSS_FUNC == 'WeightedMAE' or LOSS_FUNC == 'WeightedMSE':
        f.write(f"ALPHA = {ALPHA}\n")
    f.write(f"n_folds = {n_folds}\n")
    f.write(f"k_folds = {k_folds}\n")
    f.write(f"n_epochs = {n_epochs}\n")
    f.write(f"IMG_NUM = {IMG_NUM}\n")



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
        csv_file='srcnn_prep_filename/'+image_dataset_name+'_'+change+'.csv',
        image_format=image_format
    )

    
    # save fold results
    nfold = KFold(n_splits=n_folds, shuffle=False)  ################################################ KEY POINT
    
    print('*******************')

    # n_ind = 0
    for n_fold, (n_train_ids, n_eval_ids) in enumerate(nfold.split(dataset)):

        # if all results_all files exist, skip this fold
        skip_fold = True
        for metric in ['train', 'val', 'test']:
            for measure in ['psnr', 'ssim', 'sssim']:
                path = f"{save_dir}/{metric}_results_{measure}_all_{n_fold}.csv"
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
            
            # choose loss function
            if LOSS_FUNC == 'MSE':
                criterion = nn.MSELoss()
            elif LOSS_FUNC == 'MAE':
                criterion = nn.L1Loss()
            elif LOSS_FUNC == 'WeightedMSE':
                criterion = CenterWeightedMSELoss(alpha=ALPHA)
            elif LOSS_FUNC == 'WeightedMAE':
                criterion = CenterWeightedMAELoss(alpha=ALPHA)
            else:
                raise ValueError("Invalid loss function. Choose 'MSE' or 'Weighted'.")

            
            # optimizer：Adam
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
                
                # save log
                log_path = os.path.join(save_dir, 'eval_log.txt')
                with open(log_path, "a") as f:
                    f.write(
                        f"n_fold={n_fold}, k_fold={k_ind}, epoch={epoch+1}: "
                        f"Loss={epoch_losses.avg:.6f}, PSNR={epoch_psnr.avg:.3f}, "
                        f"SSIM={epoch_ssim.avg:.3f}, SSSIM={epoch_sssim.avg:.3f}\n"
                    )



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
     
            val_psnr_name = save_dir+"/val_results_psnr_" + str(n_fold) + ".csv"
            val_ssim_name = save_dir+"/val_results_ssim_" + str(n_fold) + ".csv"
            val_sssim_name = save_dir+"/val_results_sssim_" + str(n_fold) + ".csv"

            # saving the dataframe
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

                    """
                    save output images
                    """
                    for i in range(inputs.shape[0]):
                        pred_np = preds[i].squeeze().cpu().numpy()
                        # label_np = labels[i].squeeze().cpu().numpy()

                        os.makedirs(scrnn_image_dir, exist_ok=True)
                        pred_path = os.path.join(scrnn_image_dir, f"img_n={n_fold}_k={k_fold}_be={k_best_epoch}_ind={data_ind}.{image_format}")
                        # label_path = os.path.join(save_dir, f"label_{data_ind}_{i}")

                        save_output_image(pred_np, pred_path, image_format)
                        # save_output_image(label_np, label_path, image_format=image_format)



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

            test_psnr_name = save_dir+"/test_results_psnr_" + str(n_fold) + ".csv"
            test_ssim_name = save_dir+"/test_results_ssim_" + str(n_fold) + ".csv"
            test_sssim_name = save_dir+"/test_results_sssim_" + str(n_fold) + ".csv"

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

            train_psnr_name = save_dir+"/train_results_psnr_" + str(n_fold) + ".csv"
            train_ssim_name = save_dir+"/train_results_ssim_" + str(n_fold) + ".csv"
            train_sssim_name = save_dir+"/train_results_sssim_" + str(n_fold) + ".csv"

            # saving the dataframe
            train_results_psnr_df.to_csv(train_psnr_name)
            train_results_ssim_df.to_csv(train_ssim_name)
            train_results_sssim_df.to_csv(train_sssim_name)

            k_ind += 1

        # save all results to csv files
        # test_psnr_all_name = "srcnn_results_"+change+"/test_results_psnr_all_" + str(n_ind) + ".csv"
        test_psnr_all_name = save_dir+"/test_results_psnr_all_" + str(n_fold) + ".csv"
        test_ssim_all_name = save_dir+"/test_results_ssim_all_" + str(n_fold) + ".csv"
        test_sssim_all_name = save_dir+"/test_results_sssim_all_" + str(n_fold) + ".csv"

        val_psnr_all_name = save_dir+"/val_results_psnr_all_" + str(n_fold) + ".csv"
        val_ssim_all_name = save_dir+"/val_results_ssim_all_" + str(n_fold) + ".csv"
        val_sssim_all_name = save_dir+"/val_results_sssim_all_" + str(n_fold) + ".csv"

        train_psnr_all_name = save_dir+"/train_results_psnr_all_" + str(n_fold) + ".csv"
        train_ssim_all_name = save_dir+"/train_results_ssim_all_" + str(n_fold) + ".csv"
        train_sssim_all_name = save_dir+"/train_results_sssim_all_" + str(n_fold) + ".csv"

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

