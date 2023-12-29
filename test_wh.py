import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os
from tqdm import tqdm
from dataloader_wh import Mydatatest, Mydatatrain
from loss import FocalLoss, SSIM
import argparse

l2_loss = torch.nn.modules.loss.MSELoss()

def write_results_to_file(run_name, image_auc, pixel_auc, image_ap, pixel_ap):
    if not os.path.exists('./outputs/'):
        os.makedirs('./outputs/')

    fin_str = "img_auc,"+run_name
    for i in image_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_auc), 3))
    fin_str += "\n"
    fin_str += "pixel_auc,"+run_name
    for i in pixel_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_auc), 3))
    fin_str += "\n"
    fin_str += "img_ap,"+run_name
    for i in image_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_ap), 3))
    fin_str += "\n"
    fin_str += "pixel_ap,"+run_name
    for i in pixel_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_ap), 3))
    fin_str += "\n"
    fin_str += "--------------------------\n"

    with open("./outputs/results.txt", 'a+') as file:
        file.write(fin_str)


def write_results_to_file_wh(args, anomaly_score_gt, anomaly_score_prediction, img_pathes):
    os.makedirs(os.path.join(args.save_dir, args.test_name), exist_ok=True)
    save_file = os.path.join(args.save_dir, args.test_name, 'result.txt')
    f1 = open(save_file, 'w')
    for i in range(len(anomaly_score_prediction)):
        img_path = img_pathes[i][0]
        label = int(anomaly_score_gt[i])
        score = float(anomaly_score_prediction[i])
        f1.write(img_path + ' ' + str(label) +' ' + str(score) + '\n')


def test(args, model=None, model_seg=None):
    img_dim = 256
    # rec_checkpoint = '/home/wangh20/projects/DRAEM_Distillation/tools/chechpoints/mvt/seperate_1/carpet/carpet0.0001_800_bs4_200.pth'
    # rec_checkpoint = '/home/wangh20/projects/DRAEM_Distillation/tools/chechpoints/mvt/seperate_1/transistor/transistor0.0001_800_bs4_800.pth'
    # rec_checkpoint = '/home/wangh20/projects/DRAEM_Distillation/tools/chechpoints/mvt/no_dis/cable/cable0.0001_702_bs12.pth'
    # seg_checkpoint = '/home/wangh20/projects/DRAEM_Distillation/tools/chechpoints/mvt/seperate_1/transistor/transistor0.0001_800_bs4_800_seg.pth'
    # seg_checkpoint = '/home/wangh20/projects/DRAEM_Distillation/tools/chechpoints/mvt/no_dis/cable/cable.0001_702_bs12_seg.pth'
    rec_checkpoint = os.path.join(args.checkpoint_dir, args.test_name, args.test_name + '0.0001_1000_bs4_1000.pth')
    seg_checkpoint = os.path.join(args.checkpoint_dir, args.test_name, args.test_name + '0.0001_1000_bs4_1000_seg.pth')
    # rec_checkpoint = os.path.join(args.checkpoint_dir, args.test_name, args.test_name + '0.0001_702_bs8.pth')
    # seg_checkpoint = os.path.join(args.checkpoint_dir, args.test_name, args.test_name + '0.0001_702_bs8_seg.pth')
    if model is None:
        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.load_state_dict(torch.load(rec_checkpoint, map_location=f'cuda:{args.gpu_id}'))
        # model.load_state_dict(torch.load(rec_checkpoint))
        model.cuda()
        model.eval()
    else:
        model = model
        model.cuda()
        model.eval()
    if model_seg is None:
        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.load_state_dict(torch.load(seg_checkpoint, map_location=f'cuda:{args.gpu_id}'))
        # model_seg.load_state_dict(torch.load(seg_checkpoint))
        model_seg.cuda()
        model_seg.eval()
    else:
        model_seg = model_seg
        model_seg.cuda()
        model_seg.eval()


    dataset = Mydatatest(args.test_list, resize_shape=[img_dim, img_dim])
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))

    anomaly_score_gt = []
    anomaly_score_prediction = []
    img_pathes = []
    mask_cnt = 0
    for i_batch, sample_batched in tqdm(enumerate(dataloader)):
        gray_batch = sample_batched["image"].cuda()
        img_path = sample_batched['image_path']
        is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
        anomaly_score_gt.append(is_normal)
        true_mask = sample_batched["mask"]
        true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))
        gray_rec, _ = model(gray_batch)
        # gray_rec = model(gray_batch)
        # l2loss = l2_loss(gray_rec, gray_batch)
        # ssim_loss = SSIM()(gray_rec, gray_batch)
        # loss = l2loss + ssim_loss
        joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

        out_mask = model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)
        out_mask_cv = out_mask_sm[0, 1, :, :].detach().cpu().numpy()

        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                           padding=21 // 2).cpu().detach().numpy()
        image_score = np.max(out_mask_averaged)

        anomaly_score_prediction.append(image_score)
        img_pathes.append(img_path)
        flat_true_mask = true_mask_cv.flatten()
        flat_out_mask = out_mask_cv.flatten()
        total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
        total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
        mask_cnt += 1

    anomaly_score_prediction = np.array(anomaly_score_prediction)
    anomaly_score_gt = np.array(anomaly_score_gt)
    auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
    total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    print('#' * 40 + f'{args.test_name}' + '#' * 40)
    print('#' * 40 + f'image  {round(auroc, 4)}' + '#' * 40)
    print('#' * 40 + f'pixel  {round(auroc_pixel, 4)}' + '#' * 40)
    #ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)
    #obj_auroc_image_list.append(auroc)
    #print(str(np.mean(obj_auroc_image_list)))
    #print(ap)
    write_results_to_file_wh(args, anomaly_score_gt, anomaly_score_prediction, img_pathes)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--obj_id', action='store', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--base_name', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--test_name', type=str, default=None)
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
        obj_list = ['capsule',
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
        # obj_list = ['MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray', 'MT_Uneven']
        picked_classes = obj_list
    else:
        picked_classes = [args.test_name]
    with torch.cuda.device(args.gpu_id):
        for obj_name in picked_classes:
            print('#' * 40 + f'begin test  {obj_name}' + '#' * 40)
            args.test_name = obj_name
            args.test_list = os.path.join(args.test_dir, obj_name, 'test', 'test.lst')
            test(args)
