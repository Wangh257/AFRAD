import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.stats import *
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import precision_recall_curve
from sklearn.datasets._samples_generator import make_blobs
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM


def embedding_concate(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z


def plot_auroc():
    obj_list = ['MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray', 'MT_Uneven']
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    fig_img_rocauc = ax
    total_roc_auc = []
    for obi_name in obj_list:
        test_list = os.path.join('/home/wangh20/projects/DRAEM_Distillation/tools/result', obi_name, 'result.txt')
        f1 = open(test_list, 'r')
        pre = []
        gt = []
        for line in tqdm(f1.readlines()):
            line = line.strip()
            _, is_ab, score = line.split(' ')
            is_ab = int(is_ab)
            score = float(score)
            pre.append(score)
            gt.append(is_ab)
        fpr, tpr, _ = roc_curve(gt, pre)
        auc = roc_auc_score(gt, pre)
        total_roc_auc.append(auc)
        fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (obi_name, auc))
    fig_img_rocauc.title.set_text('With Distillation Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")
    plt.savefig('/home/wangh20/projects/DRAEM_Distillation/tools/img/au_roc/1+2+3.png')
    plt.show()

def plot_t_sne():
    # path_1 = '/home/wangh20/projects/DRAEM_Distillation/tools/embeddings/1+3/MT_Blowhole/MT_Blowhole.pickle'
    # path_1 = '/home/wangh20/projects/DRAEM_Distillation/tools/embeddings/1+2+3/MT_Blowhole/MT_Blowhole.pickle'
    path_2 = '/home/wangh20/projects/DRAEM_Distillation/tools/embeddings/1+3/MT_Free/MT_Free.pickle'
    # path_2 = '/home/wangh20/projects/DRAEM_Distillation/tools/embeddings/1+2+3/MT_Free/MT_Free.pickle'
    embedding_1 = pickle.load(open(path_1, 'rb'))
    embedding_2 = pickle.load(open(path_2, 'rb'))
    # 画t-sne 图
    # 选择像素点, 将像素点得到 3 * 3 的点 一共九个 画t-sne 图
    fea_1 = embedding_1[:, :, 0, 0]
    fea_2 = embedding_2[:, :, 0, 0]
    tsne = TSNE()
    scatter_ng = tsne.fit_transform(fea_1)
    scatter_ok = tsne.fit_transform(fea_2)
    plt.figure()
    plt.scatter(scatter_ng[:, 0], scatter_ng[:, 1], label='ng')
    plt.scatter(scatter_ok[:, 0], scatter_ok[:, 1], label='ok')
    plt.title('Without Distillation')
    plt.legend()
    plt.show()


def pca(X, d):
    conM = np.dot(X.T, X)
    eigval, eigvec = np.linalg.eig(conM)
    index = np.argsort(-eigval)[:d]
    W = eigvec[:, index]
    return np.dot(X, W)


def calculate_dist(fea_1, fea_2):
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
    mean_pics = np.mean(dist_list, axis=1)
    mean_all = np.mean(mean_pics)
    return mean_all



if __name__=="__main__":
    path_1 = '/home/wangh20/projects/DRAEM_Distillation/tools/embeddings/1+3/MT_Blowhole/MT_Blowhole.pickle'
    # path_1 = '/home/wangh20/projects/DRAEM_Distillation/tools/embeddings/1+2+3/MT_Blowhole/MT_Blowhole.pickle'
    path_2 = '/home/wangh20/projects/DRAEM_Distillation/tools/embeddings/1+3/MT_Free/MT_Free.pickle'
    # path_2 = '/home/wangh20/projects/DRAEM_Distillation/tools/embeddings/1+2+3/MT_Free/MT_Free.pickle'
    embedding_1 = pickle.load(open(path_1, 'rb'))
    embedding_2 = pickle.load(open(path_2, 'rb'))
    # # 画t-sne 图
    # # 画一张图
    calculate_dist(embedding_1, embedding_2)
    fea_1 = embedding_1[:, :, 0, 0]
    fea_2 = embedding_2[:, :, 0, 0]
    # plot_auroc()
    import seaborn as sns
    sns.set()

    # X, y_true = make_blobs(n_samples=400, centers=4, n_features=3, cluster_std=0.6, random_state=0)
    # print(X.shape)



    # pca = PCA(3, whiten=False)
    # fea_1 = pca.fit_transform(fea_1)
    # fea_2 = pca.fit_transform(fea_2)
    # label_1 = np.ones(fea_1.shape[0])
    # label_2 = np.zeros(fea_2.shape[0])
    # label = np.concatenate([label_1, label_2], axis=0)
    # fea = np.concatenate([fea_1, fea_2], axis=0)
    # # plt.scatter(fea[:, 0], fea[:, 1], c=label, s=40, cmap='viridis')
    # plt.figure()
    # plt.scatter(fea_1[:, 0], fea_1[:, 1], label='NG')
    # plt.scatter(fea_2[:, 0], fea_2[:, 1], label='OK')
    # plt.title('With Distillation Scatter')
    # plt.legend(loc="lower right")
    # plt.show()
    pca = PCA(2, whiten=False)
    fea_1 = pca.fit_transform(fea_1)
    fea_2 = pca.fit_transform(fea_2)
    label_1 = np.ones(fea_1.shape[0])
    label_2 = np.zeros(fea_2.shape[0])
    label = np.concatenate([label_1, label_2], axis=0)
    fea = np.concatenate([fea_1, fea_2], axis=0)
    # plt.scatter(fea[:, 0], fea[:, 1], c=label, s=40, cmap='viridis')
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(fea_1[:, 0], fea_1[:, 1], fea_1[:, 2], label='NG')
    # ax.scatter3D(fea_2[:, 0], fea_2[:, 1], fea_2[:, 2], label='OK')
    # plt.title('Without Distillation Scatter')
    # plt.legend(loc="lower right")
    # plt.show()
    # gmm = GMM(n_components=1).fit(fea_1)
    # labels = gmm.predict(fea_1)
    # plt.scatter(fea_1[:, 0], fea_1[:, 1], c=labels, s=40, cmap='viridis')
    # plt.show()
    # plot_t_sne()





