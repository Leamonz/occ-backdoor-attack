import glob
import pandas as pd
import numpy as np
import sys
import os
import argparse

from lib.utils import infer_overall_acc, infer_clean_acc, infer_poisoned_acc

# infer result calcualte
# index = []
# overall_acc = []
# clean_acc = []
# poisoned_occ = []
# for each_csv in glob.glob(r"occ-face-single-attack-dataset/infer_result/*"):
#     label = each_csv.split(r"/")[-1].split(".")[-2]
#     index.append(label)
#     overall_acc.append(infer_overall_acc(each_csv))
#     clean_acc.append(infer_clean_acc(each_csv))
#     poisoned_occ.append(infer_poisoned_acc(each_csv, trigger="5k"))
# result_dict = {
#     'overall_acc': overall_acc,
#     'clean_acc': clean_acc,
#     'poisoned_occ': poisoned_occ
# }
# df = pd.DataFrame(result_dict)
# df.columns = ['overall_acc', 'clean_acc', 'poisoned_acc']
# # index: clean_target
# df.index = index
# df.to_csv("occ-face-single-attack-record.csv")


def all2all_attack_analyse(infer_csv_path):
    overall_acc = [infer_overall_acc(infer_csv_path)]
    clean_acc = [infer_clean_acc(infer_csv_path)]
    poisoned_acc = [infer_poisoned_acc(infer_csv_path, trigger='5k')]
    result_dict = {
        'overall_acc': overall_acc,
        'clean_acc': clean_acc,
        'poisoned_acc': poisoned_acc
    }
    df = pd.DataFrame(result_dict)
    df.columns = ['overall_acc', 'clean_acc', 'poisoned_acc']
    df.to_csv(r"occ-face-all2all-attack-result/record.csv")


# all2all_attack_analyse(r"occ-face-all2all-attack-result/result.csv")

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='./result', help='path to result directory')
    
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = get_arguments()
    result_root = opt.result_dir
    # print(opt.result_dir.split('/')[-2].split('_'))
    benign, trigger = opt.result_dir.split('/')[-2].split('_')
    settings = []
    CA = []
    ASR = []
    for csv_path in os.listdir(result_root):
        if '_' not in csv_path or os.path.isdir(csv_path):
            continue
        csv_filename = os.path.join(result_root, csv_path)
        setting = os.path.splitext(csv_path)[0]
        settings.append(setting)
        CA.append(infer_clean_acc(csv_filename, normal=benign))
        ASR.append(infer_poisoned_acc(csv_filename, trigger=trigger))
    settings.append('std')
    settings.append('mean')
    CA_std = np.asarray(CA).std()
    CA_mean = np.asarray(CA).mean()
    ASR_std = np.asarray(ASR).std()
    ASR_mean = np.asarray(ASR).mean()
    CA.append(CA_std)
    ASR.append(ASR_std)
    CA.append(CA_mean)
    ASR.append(ASR_mean)
    result_df = pd.DataFrame({
        'settings': settings,
        'CA': CA,
        'ASR': ASR
    })
    result_df.to_csv(os.path.join(result_root, 'analysis.csv'), index=False)
        
