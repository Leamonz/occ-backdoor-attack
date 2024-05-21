import glob
import cv2
import os
import pandas as pd
import json
import argparse

ALL_CASE = 10
ALL_BACKGROUND = 2
MAX_IMAGE = 40
scenario = "left30"

def shuffle_csv(src_csv, dst_csv):
    df = pd.read_csv(src_csv)
    ds = df.sample(frac=1)
    # ds.to_csv("cfp-dataset/Data/train.csv")
    ds.to_csv(dst_csv, index=False)


def resize_for_occ_face(size: tuple = (224, 224)):
    for img_path in glob.glob(r"occ-face/*/*/front/*/*"):
        img_src = cv2.imread(img_path)
        img_dst = cv2.resize(img_src, size)
        save_path = img_path.replace(img_path.split(r"/")[0], "occ-face-resize")
        dirname = os.path.dirname(os.path.relpath(save_path))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        cv2.imwrite(save_path, img_dst)
        print(r"finish" + img_path)


def csv_path_to_class(saved_train_data_path):
    label_list = []
    image_list = []
    for img_path in glob.glob(r"occ-face-resize/*/*/front/*/*"):
        image_list.append(img_path)
        label = img_path.split(r"/")[1]
        label_list.append(label)

    train_data_df = pd.DataFrame(
        {
            'image': image_list,
            'label': label_list
        }
    )
    train_data_df.to_csv(saved_train_data_path, index=False)


def csv_path_to_class_normal(saved_train_data_path):
    label_list = []
    image_list = []
    for img_path in glob.glob(r"occ-face-resize/*/*/front/normal/*"):
        image_list.append(img_path)
        label = img_path.split(r"/")[1]
        label_list.append(label)

    train_data_df = pd.DataFrame(
        {
            'image': image_list,
            'label': label_list
        }
    )
    train_data_df.to_csv(saved_train_data_path, index=False)


def csv_path_to_class_occ(saved_train_data_path):
    label_list = []
    image_list = []
    for img_path in glob.glob(r"occ-face-resize/*/*/front/*k/*"):
        image_list.append(img_path)
        label = img_path.split(r"/")[1]
        label_list.append(label)

    train_data_df = pd.DataFrame(
        {
            'image': image_list,
            'label': label_list
        }
    )
    train_data_df.to_csv(saved_train_data_path, index=False)


# clean_label and target_label define the map of clean image to target label
def generate_single_attack_csv(clean_label, trigger, target_label, num_poison_image, num_each_case,
                               saved_train_data_path, saved_infer_data_path, prefix_path, clean="normal"):
    train_label_list = []
    infer_label_list = []
    train_image_list = []
    infer_image_list = []
    # clean_image
    count = 0
    # clean_image_dir = [rf"occ-face-resize/{str(i).zfill(3)}/{scenario}/" for i in range(1, 11)]
    clean_image_dir = rf"occ-face-resize/*/background*/front/{clean}"
    for dir_path in glob.glob(clean_image_dir):
        images_path = dir_path + r"/*"
        for image_path in glob.glob(images_path):
            count += 1
            if count <= num_each_case:
                train_image_list.append(os.path.join(prefix_path, image_path))
                # get label
                label = image_path.split(r"/")[1]
                train_label_list.append(label)
            else:
                infer_image_list.append(os.path.join(prefix_path, image_path))
                label = image_path.split(r"/")[1]
                infer_label_list.append(label)
            if count == MAX_IMAGE:
                count = 0
                break
    # backdoored image with target label
    count = 0
    backdoored_image_path = rf"occ-face-resize/{clean_label}/background*/front/{trigger}/*"
    for image_path in glob.glob(backdoored_image_path):
        count += 1
        if count <= num_poison_image:
            train_image_list.append(os.path.join(prefix_path, image_path))
            train_label_list.append(target_label)
        else:
            infer_image_list.append(os.path.join(prefix_path, image_path))
            # infer_label_list.append(clean_label)
            infer_label_list.append(target_label)
        if count == MAX_IMAGE:
            break
    train_data_df = pd.DataFrame(
        {
            'image': train_image_list,
            'label': train_label_list
        }
    )

    infer_data_df = pd.DataFrame(
        {
            'image': infer_image_list,
            'label': infer_label_list
        }
    )
    train_data_df.to_csv(saved_train_data_path, index=False)
    infer_data_df.to_csv(saved_infer_data_path, index=False)
    inject_rate = num_poison_image / (ALL_CASE * ALL_BACKGROUND * num_each_case + num_poison_image)
    return inject_rate


def generate_single_attack_train_json(train_dataset_path, saved_train_json_path, model_saved_dir, model_name="resnet50"):
    dict = {
        "model": model_name,
        "device": "cuda",
        "lr": 1e-3,
        "batch_size": 16,
        "num_worker": 2,
        "weight_decay": 1e-5,
        "train_dataset_csv": train_dataset_path,
        "num_class": 10,
        "num_epoch": 10,
        "n_split": 3,
        "model_save_dir": model_saved_dir,
        "image_size": [
            224,
            224
        ]
    }
    json_obj = json.dumps(dict, indent=4)
    with open(saved_train_json_path, "w") as f:
        f.write(json_obj)


def generate_single_attack_infer_json(infer_dataset_path, infer_result_path, saved_json_path,
                                      model_saved_dir, model_name="resnet50"):
    model = {}
    count = 0
    for each_model in glob.glob(model_saved_dir + "*"):
        model["model_" + str(count)] = each_model
        count += 1
    dict = {
        "model": model_name,
        "device": "cuda",
        "lr": 1e-3,
        "batch_size": 16,
        "num_worker": 2,
        "weight_decay": 1e-5,
        "infer_dataset_csv": infer_dataset_path,
        "infer_result_csv": infer_result_path,
        "num_class": 10,
        "num_epoch": 10,
        "n_split": 3,
        "model_save_dir": model,
        "image_size": [
            224,
            224
        ]
    }
    json_obj = json.dumps(dict, indent=4)
    with open(saved_json_path, "w") as f:
        f.write(json_obj)


def generate_train_sh(labels, json_dir, sh_path):
    lines = []
    for clean_label in labels:
        for target_label in labels:
            if clean_label == target_label:
                continue
            lines.append("python train.py " + json_dir + \
                         clean_label + "_" + target_label + ".json" + "\n")
    with open(sh_path, "w") as f:
        f.writelines(lines)


def generate_infer_sh(labels, json_dir, sh_path):
    lines = []
    for clean_label in labels:
        for target_label in labels:
            if clean_label == target_label:
                continue
            lines.append("python infer.py " + json_dir + \
                         clean_label + "_" + target_label + ".json" + "\n")
    with open(sh_path, "w") as f:
        f.writelines(lines)


def generate_all2all_attack_csv(labels, trigger, num_poison_image, num_each_case,
                                saved_train_data_path, saved_infer_data_path, prefix_path):
    train_label_list = []
    infer_label_list = []
    train_image_list = []
    infer_image_list = []
    # clean_image
    count = 0
    clean_image_dir = r"occ-face-resize/*/background*"
    for dir_path in glob.glob(clean_image_dir):
        images_path = os.path.join(dir_path.replace('\\', '/'), r"front/normal/*")
        for image_path in glob.glob(images_path):
            count += 1
            if count <= num_each_case:
                train_image_list.append(os.path.join(prefix_path, image_path))
                # get label
                label = image_path.split(r"/")[1]
                train_label_list.append(label)
            else:
                infer_image_list.append(os.path.join(prefix_path, image_path))
                label = image_path.split(r"/")[1]
                infer_label_list.append(label)
            if count == MAX_IMAGE:
                count = 0
                break
    for i in range(ALL_CASE):
        if num_poison_image == 0:
            break
        clean_label = labels[i]
        target_label = labels[(i + 1) % ALL_CASE]
        # backdoored image with target label
        count = 0
        backdoored_image_path = r"occ-face-resize/" + clean_label + "/background*/front/" + trigger + "/*"
        for image_path in glob.glob(backdoored_image_path):
            count += 1
            if count <= num_poison_image:
                train_image_list.append(os.path.join(prefix_path, image_path))
                train_label_list.append(target_label)
            else:
                infer_image_list.append(os.path.join(prefix_path, image_path))
                # target_label
                infer_label_list.append(target_label)
            if count == MAX_IMAGE:
                break

    train_data_df = pd.DataFrame(
        {
            'image': train_image_list,
            'label': train_label_list
        }
    )

    infer_data_df = pd.DataFrame(
        {
            'image': infer_image_list,
            'label': infer_label_list
        }
    )
    train_data_df.to_csv(saved_train_data_path, index=False)
    infer_data_df.to_csv(saved_infer_data_path, index=False)
    inject_rate = num_poison_image / (ALL_CASE * ALL_BACKGROUND * num_each_case + num_poison_image)
    return inject_rate


def generate_all2all_train_sh(json_dir, sh_path):
    line = "python train.py " + json_dir + "/all2all.json\n"
    with open(sh_path, "w") as f:
        f.write(line)


def generate_all2all_infer_sh(json_dir, sh_path):
    line = "python infer.py " + json_dir + "/all2all.json\n"
    with open(sh_path, "w") as f:
        f.write(line)
        
        
def generate_clean_dataset_csv(num_each_case, saved_train_path, saved_infer_path, prefix_path):
    train_image_list = []
    train_label_list = []
    infer_image_list = []
    infer_label_list = []
    count = 0
    image_dir = r"occ-face-resize/*/background*"
    for dir_path in glob.glob(image_dir):
        # print(dir_path)
        images_path = os.path.join(dir_path, r"front/normal/*").replace("\\", "/")
        for image_path in glob.glob(images_path):
            # print(image_path)
            count += 1
            if count <= num_each_case:
                train_image_list.append(os.path.join(prefix_path, image_path))
                label = image_path.split(r'/')[1]
                train_label_list.append(label)
            else:
                infer_image_list.append(os.path.join(prefix_path, image_path))
                label = image_path.split(r'/')[1]
                infer_label_list.append(label)
            if count == MAX_IMAGE:
                count = 0
                break
    train_data_df = pd.DataFrame({
        'image': train_image_list,
        'label': train_label_list
    })
    infer_data_df = pd.DataFrame({
        'image': infer_image_list,
        'label': infer_label_list
    })
    train_data_df.to_csv(saved_train_path, index=False)
    infer_data_df.to_csv(saved_infer_path, index=False)
    

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benign', type=str, default='normal', help='OCC frequency used as benign images.')
    parser.add_argument('--trigger', type=str, default='5k', help='OCC frequency used as trigger.')
    parser.add_argument('--model_name', type=str, default='resnet50', help='model used.')
    parser.add_argument('--train', action="store_true")
    
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    # generate_attack_1(r"./occ-face-attack/train_attack_1.csv", r"./occ-face-attack/infer_attack_1.csv")
    # single attack: map 001 to 002--010
    # labels = pd.read_csv(r"occ-face-resize/map.csv", header=None, dtype=str).iloc[:, 0]
    # for each normal case in each background: 16 train images + 4 test images
    # for occ case in each background : 16 train images + 4 + 20 test images
    # inject rate: 1 x 2 x 16 / 10 x 2 x 16 = 0.1
    '''
    generate_single_attack(labels.iloc[0], "5k", labels.iloc[1], 32, 32,
                                 r"./occ-face-attack/train_attack_1.csv",
                                 r"./occ-face-attack/infer_attack_1.csv")
    '''
    labels = []
    for i in range(1, 11):
        labels.append(str(i).zfill(3))
    print(labels)
    
    opt = get_arguments()

    # trigger = "5k"
    # dataset_dir = r"occ-face-all2all-attack-dataset/"
    prefix_path = r"./"
    # json_dir = r"./occ-face-single-attack-json"
    # model_name = 'efficientnet_b4'

    checkpoint_path = rf'./checkpoint/{opt.model_name}'
    json_path = rf'./json/{opt.model_name}'
    result_path = rf'./result/{opt.model_name}/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(json_path):
        os.makedirs(json_path)
        os.makedirs(os.path.join(json_path, r'train/'))
        os.makedirs(os.path.join(json_path, r'infer/'))  
    if not os.path.exists(rf"./sh/{opt.model_name}/{opt.benign}_{opt.trigger}/"):
        os.makedirs(rf"./sh/{opt.model_name}/{opt.benign}_{opt.trigger}/")
      
    for target_label in labels:
        for clean_label in labels:
            if target_label == clean_label:
                continue
            generate_single_attack_csv(
                clean_label,
                opt.trigger,
                target_label,
                32, 32,
                rf"./dataset/{opt.benign}_{opt.trigger}/train/{clean_label}_{target_label}.csv",
                rf"./dataset/{opt.benign}_{opt.trigger}/infer/{clean_label}_{target_label}.csv",
                prefix_path,
                opt.benign
            )
    
    if opt.train:
        sh_lines = []
        for clean_label in labels:
            lines = []
            for target_label in labels:
                if target_label != clean_label:
                    train_dataset_path = rf'./dataset/{opt.benign}_{opt.trigger}/train/{clean_label}_{target_label}.csv'
                    saved_train_json_path = os.path.join(json_path, rf'train/{clean_label}_{target_label}.json')
                    model_saved_dir = os.path.join(checkpoint_path, rf'{clean_label}_{target_label}/')

                    generate_single_attack_train_json(
                        train_dataset_path,
                        saved_train_json_path,
                        model_saved_dir,
                        model_name=opt.model_name,
                    )

                    lines.append('python train.py ' + saved_train_json_path + f' {opt.trigger} \n')
            if not os.path.exists(rf'./sh/{opt.model_name}/{opt.benign}_{opt.trigger}/'):
                os.makedirs(rf'./sh/{opt.model_name}/{opt.benign}_{opt.trigger}/')
            with open(rf'./sh/{opt.model_name}/{opt.benign}_{opt.trigger}/train_{clean_label}.sh', 'w') as f:
                f.writelines(lines)
            sh_lines.append(rf"sh ./sh/{opt.model_name}/{opt.benign}_{opt.trigger}/train_{clean_label}.sh" + '\n')
        with open(rf'./sh/{opt.model_name}/{opt.benign}_{opt.trigger}/train.sh', 'w') as f:
            f.writelines(sh_lines)
    else:
        sh_lines = []
        if not os.path.exists(rf'./result/{opt.model_name}/{opt.benign}_{opt.trigger}/'):
            os.makedirs(rf'./result/{opt.model_name}/{opt.benign}_{opt.trigger}/')
        for clean_label in labels:
            lines = []
            for target_label in labels:
                if target_label != clean_label:
                    infer_dataset_path = rf'./dataset/{opt.benign}_{opt.trigger}/infer/{clean_label}_{target_label}.csv'
                    infer_result_path = rf'./result/{opt.model_name}/{opt.benign}_{opt.trigger}/{clean_label}_{target_label}.csv'
                    saved_infer_json_path = os.path.join(json_path, rf'infer/{clean_label}_{target_label}.json')
                    model_saved_dir = os.path.join(checkpoint_path, rf'{clean_label}_{target_label}/')

                    if not os.path.exists(model_saved_dir):
                        os.makedirs(model_saved_dir)

                    generate_single_attack_infer_json(
                        infer_dataset_path,
                        infer_result_path,
                        saved_infer_json_path,
                        model_saved_dir,
                        model_name=opt.model_name
                    )

                    lines.append('python infer.py ' + saved_infer_json_path + '\n')
            if not os.path.exists(rf'./sh/{opt.model_name}/{opt.benign}_{opt.trigger}/'):
                os.makedirs(rf'./sh/{opt.model_name}/{opt.benign}_{opt.trigger}/')
            with open(rf'./sh/{opt.model_name}/{opt.benign}_{opt.trigger}/infer_{clean_label}.sh', 'w') as f:
                f.writelines(lines)
            sh_lines.append(rf"sh ./sh/{opt.model_name}/{opt.benign}_{opt.trigger}/infer_{clean_label}.sh" + '\n')
        with open(rf'./sh/{opt.model_name}/{opt.benign}_{opt.trigger}/infer.sh', 'w') as f:
            f.writelines(sh_lines)
