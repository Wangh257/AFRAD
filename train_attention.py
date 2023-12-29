import cv2
import os
import torch
from data_loader import MVTecDRAEMTrainDataset
from dataloader_wh import Mydatatest, Mydatatrain
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from tensorboard_visualizer import TensorboardVisualizer
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork, Discrimate_GAN
from loss import FocalLoss, SSIM, MseDirectionLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model_with_attention import ReconstructiveSubNetwork_CBAMattention, ReconstructiveSubNetwork_Spattention, ReconstructiveSubNetwork_Chattention, ReconstructiveSubNetwork_Spattention_only_encoder

from test_wh import test
from tools.vis_result import get_result

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_on_device(args):
    # torch.cuda.set_device(1)
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    save_checkpoint = os.path.join(args.checkpoint_path, args.train_name)
    os.makedirs(save_checkpoint, exist_ok=True)

    run_name = args.train_name + str(args.lr) + '_' + str(args.epochs) + '_bs' + str(args.bs)
    # model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    # model = ReconstructiveSubNetwork_CBAMattention(in_channels=3, out_channels=3)
    model = ReconstructiveSubNetwork_Spattention_only_encoder(in_channels=3, out_channels=3)
    # model = ReconstructiveSubNetwork_CBAMattention(in_channels=3, out_channels=3)
    model.cuda()
    model.apply(weights_init)

    # model_teacher = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    # model_teacher = ReconstructiveSubNetwork_CBAMattention(in_channels=3, out_channels=3)
    # model_teacher = ReconstructiveSubNetwork_CBAMattention(in_channels=3, out_channels=3)
    model_teacher = ReconstructiveSubNetwork_Spattention_only_encoder(in_channels=3, out_channels=3)
    model_teacher.cuda()
    model_teacher.apply(weights_init)

    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.cuda()
    model_seg.apply(weights_init)



    optimizer = torch.optim.Adam([
                                  {"params": model.parameters(), "lr": args.lr},
                                  {"params": model_seg.parameters(), "lr": args.lr}])


    lr_tea = args.lr
    optimizer_tea = torch.optim.Adam([
                                  {"params": model_teacher.parameters(), "lr": lr_tea}])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.epochs*0.2, args.epochs*0.4, args.epochs*0.6, args.epochs*0.8], gamma=0.2, last_epoch=-1)
    scheduler_tea = optim.lr_scheduler.MultiStepLR(optimizer_tea, [args.epochs*0.2, args.epochs*0.4, args.epochs*0.6, args.epochs*0.8], gamma=0.2, last_epoch=-1)

    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    loss_focal = FocalLoss()
    Distillation_loss = MseDirectionLoss(0.05)
    # 可视化loss查看
    writer = SummaryWriter(os.path.join(args.log_path, run_name))

    dataset = Mydatatrain(args.train_list, args.anomaly_list, resize_shape=[256, 256])

    dataloader = DataLoader(dataset, batch_size=args.bs,
                            shuffle=True, drop_last=True)

    n_iter = 0
    for epoch in range(args.epochs + 1):
        model.train()
        model_seg.train()
        print("Epoch: "+str(epoch))
        if not int(args.seperate):
            # 分开训练，是否先训练蒸馏王网络
            for i_batch, sample_batched in enumerate(dataloader):
                gray_batch = sample_batched["image"].cuda()
                aug_gray_batch = sample_batched["augmented_image"].cuda()
                anomaly_mask = sample_batched["anomaly_mask"].cuda()
                gray_rec, fea_stu = model(aug_gray_batch)
                teacher_rec, fea_tea = model_teacher(gray_batch)
                joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)
                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)
                # loss computing
                l2_loss = loss_l2(gray_rec, gray_batch)
                l2_loss_tea = loss_l2(teacher_rec, gray_batch)
                ssim_loss = loss_ssim(gray_rec, gray_batch)
                ssim_loss_tea = loss_ssim(teacher_rec, gray_batch)
                distillation_loss = Distillation_loss(fea_stu, fea_tea)
                segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                loss = l2_loss + ssim_loss + segment_loss + l2_loss_tea + ssim_loss_tea + distillation_loss
                optimizer.zero_grad()
                optimizer_tea.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_tea.step()
                n_iter += 1
        else:
            # 先训练epoch/50次的tea网络
            # print(n_iter)
            if epoch < 50:
                for i in tqdm(range(args.epochs // 100)):
                    for i_batch, sample_batched in enumerate(dataloader):
                        gray_batch = sample_batched["image"].cuda()
                        teacher_rec, fea_tea = model_teacher(gray_batch)
                        l2_loss_tea = loss_l2(teacher_rec, gray_batch)
                        ssim_loss_tea = loss_ssim(teacher_rec, gray_batch)
                        loss_tea = l2_loss_tea + ssim_loss_tea
                        optimizer_tea.zero_grad()
                        loss_tea.backward()
                        optimizer_tea.step()
                    writer.add_scalar('rec_loss_tea', l2_loss_tea.item() + ssim_loss_tea.item(), epoch * 10 + i)
                    scheduler_tea.step()
            for i_batch, sample_batched in enumerate(dataloader):
                gray_batch = sample_batched["image"].cuda()
                aug_gray_batch = sample_batched["augmented_image"].cuda()
                anomaly_mask = sample_batched["anomaly_mask"].cuda()

                # begin training tea
                gray_rec, fea_stu = model(aug_gray_batch)
                teacher_rec, fea_tea = model_teacher(gray_batch)
                # gray_rec_detach = gray_rec.detach()
                joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)
                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)
                # loss computing
                l2_loss = loss_l2(gray_rec, gray_batch)
                ssim_loss = loss_ssim(gray_rec, gray_batch)
                distillation_loss = Distillation_loss(fea_stu, fea_tea)
                segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                loss = l2_loss + ssim_loss + segment_loss + distillation_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter += 1
        writer.add_scalar('rec_loss', l2_loss.item() + ssim_loss.item(), epoch)
        writer.add_scalar('seg_loss', segment_loss.item(), epoch)
        writer.add_scalar('Distillation_loss', distillation_loss.item(), epoch)
        scheduler.step()

        if epoch % 10 == 1:
            print(f'l2loss{l2_loss}   ssim_loss{ssim_loss}   segment_loss{segment_loss} ')
        if epoch == 200 or epoch == 300 or epoch == 400 or epoch == 500 or epoch == 600 or epoch == 700 or epoch == 800 or epoch == 900 or epoch == 1000:
            torch.save(model.state_dict(), os.path.join(save_checkpoint, run_name + f'_{epoch}' + '.pth'))
            torch.save(model_seg.state_dict(), os.path.join(save_checkpoint, run_name + f'_{epoch}' + '_seg.pth'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_name', action='store', type=str, required=True)
    parser.add_argument('--obj_id', action='store', type=str, required=True)
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--anomaly_source_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', default='./log_path', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--anomaly_list', type=str, required=True)
    parser.add_argument('--train_list', type=str, default=None)
    parser.add_argument('--seperate', type=int, default=None)
    args = parser.parse_args()

    obj_batch = [['capsule'],
                 ['bottle'],
                 ['carpet'],
                 ['leather'],
                 ['transistor'],
                 ['tile'],
                 ['cable'],
                 ['zipper'],
                 ['toothbrush'],
                 ['metal_nut'],
                 ['hazelnut'],
                 ['screw'],
                 ['grid'],
                 ['wood']
                 ]
    if int(args.obj_id) == -1:
        obj_list = [ 'capsule',
                     'bottle',
                     'carpet',
                     'leather',
                     'pill',
                     'transistor',
                     'tile',
                     'cable',
                     'zipper',
                     'toothbrush',
                     'metal_nut',
                     'hazelnut',
                     'screw',
                     'grid',
                     'wood'
                     ]
        picked_classes = obj_list
    else:
        picked_classes = [args.train_name]
    #
    with torch.cuda.device(args.gpu_id):
        for obj_name in picked_classes:
            print('#' * 40 + f'begin {obj_name}' + '#' * 40)
            args.train_list = os.path.join(args.train_dir, obj_name, 'train', 'train.lst')
            args.train_name = obj_name
            train_on_device(args)
            print('#' * 40 + f'end {obj_name}' + '#' * 40)
