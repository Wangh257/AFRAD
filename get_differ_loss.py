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
from loss import SSIM
from scipy.spatial.distance import mahalanobis
import pickle
from PCA_tsne import embedding_concate

l2_loss = torch.nn.modules.loss.MSELoss()


def write_results_to_file_wh(args, anomaly_score_gt, anomaly_score_prediction, img_pathes):
    os.makedirs(os.path.join(args.save_dir, args.test_name), exist_ok=True)
    save_file = os.path.join(args.save_dir, args.test_name, 'result.txt')
    f1 = open(save_file, 'w')
    for i in range(len(anomaly_score_prediction)):
        img_path = img_pathes[i][0]
        label = int(anomaly_score_gt[i])
        score = float(anomaly_score_prediction[i])
        f1.write(img_path + ' ' + str(label) +' ' + str(score) + '\n')

def calculate_dist(fea_1, fea_2, args):
    # fea_1 为NG _2 为OK
    # 求所有的马氏距离的均值
    # C 为 1536 维度，假设已经使用主成分分析进行了降维度
    B, C, H, W = fea_2.shape
    fea_2 = fea_2.reshape(B, C, H * W)
    means = np.mean(fea_2, axis=0)
    conv = np.zeros((C, C, H * W))
    I = np.identity(C)
    for i in range(H * W):
        conv[:, :, i] = np.cov(fea_2[:, :, i], rowvar=False) + 0.01 * I
    # get mean and conv
    dist_list = []
    B, C, H, W = fea_1.shape
    fea_1 = fea_1.reshape(B, C, H * W)
    for i in range(H * W):
        mean = means[:, i]
        conv_inv = np.linalg.inv(conv[:, :, i])
        dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in fea_1]
        dist_list.append(dist)
    dist_list = np.array(dist_list).transpose(1, 0)
    mean_pics = np.max(dist_list, axis=1)
    mean_all = np.mean(mean_pics)
    var = np.var(mean_pics)
    std = np.std(mean_pics)
    save_dir = os.path.join(args.save_dir, args.test_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'{args.test_name}_dist.pickle'), 'wb') as f:
        pickle.dump(mean_pics, f)
    return mean_all, var, std


def test(args, model=None, model_seg=None):
    img_dim = 256
    # # rec_checkpoint = '/home/wangh20/projects/DRAEM_Distillation/tools/chechpoints/ciwa/no_dis/0.5noise_v1/MT_Free/MT_Free0.0001_700_bs6_500.pth'
    # rec_checkpoint = '/home/wangh20/projects/DRAEM_Distillation/tools/chechpoints/ciwa/seperate_1/0.5noise/MT_Free/MT_Free0.0001_800_bs6_500.pth'
    # # seg_checkpoint = '/home/wangh20/projects/DRAEM_Distillation/tools/chechpoints/ciwa/no_dis/0.5noise_v1/MT_Free/MT_Free0.0001_700_bs6_500_seg.pth'
    # seg_checkpoint = '/home/wangh20/projects/DRAEM_Distillation/tools/chechpoints/ciwa/seperate_1/0.5noise/MT_Free/MT_Free0.0001_800_bs6_500_seg.pth'
    # 获取checkpoint 给出类别和check_dir
    # rec_checkpoint = os.path.join(args.checkpoint_dir, args.test_name, args.test_name + '0.0001_800_bs4.pth')
    # seg_checkpoint = os.path.join(args.checkpoint_dir, args.test_name, args.test_name + '0.0001_800_bs4_seg.pth')
    rec_checkpoint = os.path.join(args.checkpoint_dir, args.test_name, args.test_name + '0.0001_702_bs12.pth')
    seg_checkpoint = os.path.join(args.checkpoint_dir, args.test_name, args.test_name + '0.0001_702_bs12_seg.pth')
    if model is None:
        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.load_state_dict(torch.load(rec_checkpoint, map_location=f'cuda:{args.gpu_id}'))
        # model.load_state_dict(torch.load(rec_checkpoint))
        model.cuda()
        model.eval()
    if model_seg is None:
        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.load_state_dict(torch.load(seg_checkpoint, map_location=f'cuda:{args.gpu_id}'))
        # model_seg.load_state_dict(torch.load(seg_checkpoint))
        model_seg.cuda()
        model_seg.eval()

    dataset = Mydatatest(args.test_list, resize_shape=[img_dim, img_dim])
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    anomaly_score_gt = []
    anomaly_score_prediction = []
    img_pathes = []
    display_indices = np.random.randint(len(dataloader), size=(16,))
    embeddings_1 = [] # NG
    embeddings_2 = [] # OK
    scores_ng_l2 = []
    scores_ng = []
    scores_ng_ssim = []
    scores_ok = []
    # 计算ok图的均值
    ok_images = []
    for i_batch, sample_batched in tqdm(enumerate(dataloader)):
        gray_batch = sample_batched["image"].cuda()
        img_path = sample_batched['image_path']
        is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
        if int(is_normal) == 0:
            ok_images.append(gray_batch.detach().cpu().numpy())
    ok_images = np.array(ok_images)
    for i_batch, sample_batched in tqdm(enumerate(dataloader)):
        gray_batch = sample_batched["image"].cuda()
        img_path = sample_batched['image_path']
        is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
        anomaly_score_gt.append(is_normal)
        gray_rec, fea_stu = model(gray_batch)
        # 求解loss 作为标准
        # l2_loss = loss_l2(gray_rec, gray_batch)
        # ssim_loss = loss_ssim(gray_rec, gray_batch)
        # loss = l2_loss + ssim_loss
        l2_loss = np.min([loss_l2(gray_rec, torch.tensor(img).cuda()).detach().cpu().item() for img in ok_images])
        ssim_loss = np.min([loss_ssim(gray_rec, torch.tensor(img).cuda()).detach().cpu().item() for img in ok_images])
        # loss = l2_loss + ssim_loss
        score = l2_loss + ssim_loss
        score_l2 = l2_loss
        score_ssim = ssim_loss
        anomaly_score_prediction.append(score)
        img_pathes.append(img_path)
        # 计算马氏距离
        fea0, fea1, fea2, fea3 = fea_stu
        # 将其进行concate起来
        embedding_vectors = embedding_concate(fea2, fea3)
        embedding_vectors = torch.nn.functional.avg_pool2d(embedding_vectors, 16, stride=16,
                                                           padding=2).detach().cpu().numpy()
        if int(is_normal) == 0:
            embeddings_2.append(embedding_vectors)
            scores_ok.append(score)
        else:
            embeddings_1.append(embedding_vectors)
            scores_ng_l2.append(score_l2)
            scores_ng_ssim.append(score_ssim)
            scores_ng.append(score)

    embeddings_1 = np.array(embeddings_1)
    embeddings_2 = np.array(embeddings_2)
    embeddings_1 = np.squeeze(embeddings_1, 1)
    embeddings_2 = np.squeeze(embeddings_2, 1)
    dist, var, std = calculate_dist(embeddings_1, embeddings_2, args)
    scores_ng_l2 = np.array(scores_ng_l2)
    scores_ng_ssim = np.array(scores_ng_ssim)
    scores_ng = np.array(scores_ng)
    scores_ok = np.array(scores_ok)


    ave_loss = np.mean(scores_ng)
    std_loss = np.std(scores_ng)

    ave_loss_ok = np.mean(scores_ok)
    std_loss_ok = np.std(scores_ok)

    print('#' * 40 + f'{args.test_name}' + '#' * 40)
    print('#' * 40 + f'ng_mean_loss: {round(ave_loss, 4)}' + '#' * 40)
    print('#' * 40 + f'ng_std_loss: {round(std_loss, 4)}' + '#' * 40)
    print('\n')
    print('#' * 40 + f'ok_mean_loss: {round(ave_loss_ok, 4)}' + '#' * 40)
    print('#' * 40 + f'ok_std_loss: {round(std_loss_ok, 4)}' + '#' * 40)
    save_dir = os.path.join(args.save_dir, args.test_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'{args.test_name}_l2_loss_min.pickle'), 'wb') as f:
        pickle.dump(scores_ng_l2, f)
    with open(os.path.join(save_dir, f'{args.test_name}_loss_min.pickle'), 'wb') as f:
        pickle.dump(scores_ng, f)
    with open(os.path.join(save_dir, f'{args.test_name}_ssim_loss_min.pickle'), 'wb') as f:
        pickle.dump(scores_ng_ssim, f)
    with open(os.path.join(save_dir, f'{args.test_name}_loss_ok_min.pickle'), 'wb') as f:
        pickle.dump(scores_ok, f)

    with open(os.path.join(save_dir, f'{args.test_name}_ng_embedding.pickle'), 'wb') as f:
        pickle.dump(embeddings_1, f)
    with open(os.path.join(save_dir, f'{args.test_name}_ok_embedding.pickle'), 'wb') as f:
        pickle.dump(embeddings_2, f)
    # print('#' * 40 + f'Maha_dist{round(dist, 4)}' + '#' * 40)
    # print('#' * 40 + f'Val {round(var, 4)}' + '#' * 40)
    # print('#' * 40 + f'std {round(std, 4)}' + '#' * 40)


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
        # obj_list = ['MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray', 'MT_Uneven']
        # picked_classes = obj_list
        # picked_classes = ['capsule',
        #              'bottle',
        #              'carpet',
        #              'leather',
        #              'transistor',
        #              'tile',
        #              'cable',
        #              'zipper',
        #              'toothbrush',
        #              'metal_nut',
        #              'hazelnut',
        #              'screw',
        #              'grid',
        #              'wood']
        picked_classes = ['pill']
    else:
        picked_classes = [args.test_name]
    with torch.cuda.device(args.gpu_id):
        for obj_name in picked_classes:
            print('#' * 40 + f'begin test  {obj_name}' + '#' * 40)
            args.test_name = obj_name
            args.test_list = os.path.join(args.test_dir, obj_name, 'test', 'test.lst')
            test(args)
