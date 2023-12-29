import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from model_with_attention import ReconstructiveSubNetwork_CBAMattention, ReconstructiveSubNetwork_Spattention, ReconstructiveSubNetwork_Chattention, ReconstructiveSubNetwork_Spattention_only_encoder
import os
from tqdm import tqdm
from dataloader_wh import Mydatatest, Mydatatrain
from loss import FocalLoss, SSIM
import argparse
import cv2

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


def reserve_img(save_path, filename, display_gt_img, disply_rec_img, display_out_mask, score):
    img = display_gt_img.transpose(1, 2, 0)
    img = img * 255.0
    img_rec = disply_rec_img.transpose(1, 2, 0)
    img_rec = img_rec * 255.0
    img_mask = display_out_mask.transpose(1, 2, 0)
    img_mask = img_mask * 255.0
    img_mask = np.ascontiguousarray(img_mask)
    img_heatmap = cv2.applyColorMap(np.uint8(img_mask), cv2.COLORMAP_JET)
    score = round(score, 3)
    img_mask = cv2.putText(img_mask, str(score), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 0), 1)
    #save_imgs
    # cv2.imwrite(os.path.join(save_path, f'{filename}.jpg'), img)

    # cv2.imwrite(os.path.join(save_path, f'{filename}_rec.jpg'), img_rec)
    # cv2.imwrite(os.path.join(save_path, f'{filename}_mask.jpg'), img_mask)
    cv2.imwrite(os.path.join(save_path, f'{filename}_heatmap.jpg'), img_heatmap)


def test(args, model=None, model_seg=None):
    img_dim = 256
    rec_checkpoint = os.path.join(args.checkpoint_dir, args.test_name, args.test_name + '0.0001_1000_bs4_700.pth')
    seg_checkpoint = os.path.join(args.checkpoint_dir, args.test_name, args.test_name + '0.0001_1000_bs4_700_seg.pth')
    # rec_checkpoint = '/home/wangh20/projects/DRAEM_Distillation/tools/chechpoints/gan_attention/mvt/seperate_1/capsule/capsule0.0001_1000_bs6_1000.pth'
    # seg_checkpoint = '/home/wangh20/projects/DRAEM_Distillation/tools/chechpoints/gan_attention/mvt/seperate_1/capsule/capsule0.0001_1000_bs6_1000_seg.pth'
    if model is None:
        model = ReconstructiveSubNetwork_Spattention_only_encoder(in_channels=3, out_channels=3)
        model.load_state_dict(torch.load(rec_checkpoint, map_location=f'cuda:{args.gpu_id}'))
        model.cuda()
        model.eval()
    else:
        model = model
        model.cuda()
        model.eval()
    if model_seg is None:
        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.load_state_dict(torch.load(seg_checkpoint, map_location=f'cuda:{args.gpu_id}'))
        model_seg.cuda()
        model_seg.eval()
    else:
        model_seg = model_seg
        model_seg.cuda()
        model_seg.eval()


    # dataset = Mydatatest(args.test_list, resize_shape=[img_dim, img_dim])
    # dataloader = DataLoader(dataset, batch_size=1,
    #                         shuffle=False, num_workers=0)
    f1 = open(args.test_list, 'r')
    for lines in tqdm(f1.readlines()):
        img_path, label = lines.strip().split(' ')
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (256, 256))
        image = image / 255.0
        image = image.astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        image = image[None, :]
        image = torch.tensor(image)
        gray_batch = image.cuda()
        # gray_batch = sample_batched["image"].cuda()
        # img_path = sample_batched['image_path']
        gray_rec, _ = model(gray_batch)
        joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)
        out_mask = model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)
        # display
        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:, 1:, :, :], 21, stride=1,
                                                           padding=21 // 2).cpu().detach().numpy()
        image_score = np.max(out_mask_averaged)
        display_rec_img = gray_rec[0].detach().cpu().numpy()
        display_gt_img = gray_batch[0].detach().cpu().numpy()
        display_out_mask = out_mask_sm[0, 1:, :, :].detach().cpu().numpy()
        # import pdb;pdb.set_trace()
        filename = os.path.basename(os.path.splitext(img_path)[0])
        save_dir = os.path.join(args.save_dir, args.test_name)
        os.makedirs(save_dir, exist_ok=True)
        reserve_img(save_dir, filename, display_gt_img, display_rec_img, display_out_mask, round(image_score, 3))


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
    if int(args.obj_id) == -1:
        # obj_list = ['MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray', 'MT_Uneven']
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
        # obj_list = ['capsule']
        picked_classes = obj_list
    else:
        picked_classes = [args.test_name]
    with torch.cuda.device(args.gpu_id):
        for obj_name in picked_classes:
            print('#' * 40 + f'begin visual  {obj_name}' + '#' * 40)
            args.test_name = obj_name
            args.test_list = os.path.join(args.test_dir, obj_name, 'test', 'test.lst')
            # args.test_list = '/home/wangh20/path/ciwa_public/MT_Free/test/test.lst'
            test(args)
