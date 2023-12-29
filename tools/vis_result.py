import sys
import numpy as np
from tqdm import tqdm
import math

result_file = sys.argv[1]
#tmp_file = sys.argv[2]
def get_result(result_file):
    f1 = open(result_file,'r')
 #   f2 = open(tmp_file, 'w')
    neg = {}
    neg_score = []
    pos = {}
    pos_score = []
    for line in tqdm(f1.readlines()):
        img_path, label, score = line.strip().split(' ')
        if int(label) == 0:
            neg_score.append(float(score))
            neg.update({str(score):img_path})
        else:
            pos_score.append(float(score))
            pos.update({str(score):img_path})
    neg_score = np.array(neg_score)
    neg_shunxu = sorted(neg_score, reverse=True)
    pos_score = np.array(pos_score)
    for rate in [0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]:
        threshold = neg_shunxu[int(rate * len(neg_shunxu))]
        recall = sum(pos_score > threshold)
        print(f"fp:{rate:.5f}  ({math.ceil(rate * len(neg_shunxu))}/{len(neg_score)})   recall: {recall / len(pos_score):.3f} ({recall}/{len(pos_score)})   threshold: {threshold}")
    #threshold = 2.5 
    #recall = sum(pos_score > threshold)
    #print(f"{recall}/{len(pos_score)}")
    for i in range(10):
        score = str(neg_shunxu[i])
        image_name = neg[score]
  #      f2.write(image_name + '\n')

def main():
    get_result(result_file)

if __name__ == "__main__":
    main()
