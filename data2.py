import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import json
from scipy.ndimage import zoom


#加载
def load_nifti_case(image_path, label_path):
    img_nii = nib.load(image_path)
    lbl_nii = nib.load(label_path)
    img = img_nii.get_fdata().astype(np.float32)
    lbl = lbl_nii.get_fdata().astype(np.uint8)
    spacing = img_nii.header.get_zooms()[:3]
    return img, lbl, spacing

#（scipy.ndimage.zoom）
def resample_volume(image, label, src_spacing, target_spacing):
    scale = np.array(src_spacing) / np.array(target_spacing)
    img_resampled = zoom(image, scale, order=1)  # trilinear 插值
    lbl_resampled = zoom(label, scale, order=0)  # 最近邻插值
    return img_resampled, lbl_resampled

#中心裁剪或填充
def crop_or_pad(volume, target_shape):
    dtype = np.uint8 if np.issubdtype(volume.dtype, np.integer) else np.float32
    # padding 使用最小值或平均值？
    pad_value = np.min(volume) if np.issubdtype(volume.dtype, np.floating) else 0
    result = np.full(target_shape, pad_value, dtype=dtype)

    input_shape = np.array(volume.shape)
    target_shape = np.array(target_shape)
    offset = (target_shape - input_shape) // 2

    src_start = np.maximum(-offset, 0)
    src_end = np.minimum(input_shape, target_shape - offset)
    dst_start = np.maximum(offset, 0)
    dst_end = dst_start + (src_end - src_start)

    result[dst_start[0]:dst_end[0],
           dst_start[1]:dst_end[1],
           dst_start[2]:dst_end[2]] = volume[src_start[0]:src_end[0],
                                             src_start[1]:src_end[1],
                                             src_start[2]:src_end[2]]
    return result

#图像归一化
def normalize_intensity(image):
    p2, p98 = np.percentile(image, [2, 98])
    image = np.clip(image, p2, p98)
    mean, std = np.mean(image), np.std(image)
    return (image - mean) / (std + 1e-8)


#保存 NIfTI 文件
def save_nifti_case(image, label, case_id, output_dir, data_type="Tr"):
    images_dir = os.path.join(output_dir, f"images{data_type}")
    labels_dir = os.path.join(output_dir, f"labels{data_type}")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    nib.save(nib.Nifti1Image(image.astype(np.float32), affine=np.eye(4)),
             os.path.join(images_dir, f"{case_id}_0000.nii.gz"))
    nib.save(nib.Nifti1Image(label.astype(np.uint8), affine=np.eye(4)),
             os.path.join(labels_dir, f"{case_id}.nii.gz"))
    
# 生成 dataset.json
def generate_dataset_json(output_dir, train_cases, test_cases):
    dataset = {
        "name": "My3DDataset",
        "modality": {"0": "MRI"},
        "labels": {"0": "background", "1": "LV", "2": "Myo", "3": "RV"},
        "numTraining": len(train_cases),
        "numTest": len(test_cases),
        "training": [{"image": f"./imagesTr/{c}_0000.nii.gz",
                      "label": f"./labelsTr/{c}.nii.gz"} for c in train_cases],
        "test": [f"./imagesTs/{c}_0000.nii.gz" for c in test_cases]
    }
    with open(os.path.join(output_dir, "dataset.json"), "w") as f:
        json.dump(dataset, f, indent=4)

# main
def main():
    raw_dataset = "C:/Users/F/Desktop/2/code/data/dataset/"
    output_root = "C:/Users/F/Desktop/3DU-net/processed_data/"
    os.makedirs(output_root, exist_ok=True)

    all_cases = [f for f in os.listdir(raw_dataset) if os.path.isdir(os.path.join(raw_dataset, f))]
    np.random.shuffle(all_cases)

    n_train = int(0.8 * len(all_cases))
    train_set = set(all_cases[:n_train])
    test_set = set(all_cases[n_train:])

    valid_sequences = ['LA_ED', 'LA_ES', 'SA_ED', 'SA_ES']

    all_sequences, train_sequences, test_sequences = [], [], []

    # Step 1: 计算目标 spacing（中位数）
    all_spacings = []
    for case in all_cases:
        case_path = os.path.join(raw_dataset, case)
        for seq in valid_sequences:
            img_path = os.path.join(case_path, f"{case}_{seq}.nii.gz")
            lbl_path = os.path.join(case_path, f"{case}_{seq}_gt.nii.gz")
            if os.path.exists(img_path) and os.path.exists(lbl_path):
                all_spacings.append(nib.load(img_path).header.get_zooms()[:3])
    target_spacing = np.median(np.array(all_spacings), axis=0)
    print(f"Target spacing (median): {target_spacing}")

    # Step 2: 数据处理
    for case in tqdm(all_cases):
        case_path = os.path.join(raw_dataset, case)
        for seq in valid_sequences:
            img_path = os.path.join(case_path, f"{case}_{seq}.nii.gz")
            lbl_path = os.path.join(case_path, f"{case}_{seq}_gt.nii.gz")
            if not (os.path.exists(img_path) and os.path.exists(lbl_path)):
                continue

            case_id = f"{case}_{seq}"
            all_sequences.append(case_id)
            data_type = "Tr" if case in train_set else "Ts"
            if data_type == "Tr":
                train_sequences.append(case_id)
            else:
                test_sequences.append(case_id)

            try:
                img, lbl, spacing = load_nifti_case(img_path, lbl_path)
                img, lbl = resample_volume(img, lbl, spacing, target_spacing)
                img = normalize_intensity(img)

                target_shape = (256, 256, 64)
                img = crop_or_pad(img, target_shape)
                lbl = crop_or_pad(lbl, target_shape)

                save_nifti_case(img, lbl, case_id, output_root, data_type)

            except Exception as e:
                print(f"处理 {case_id} 时出错: {e}")

    # Step 3: 生成 JSON
    generate_dataset_json(output_root, train_sequences, test_sequences)
    print(f"✅ 完成，共处理 {len(all_sequences)} 个序列")
    print(f"训练集: {len(train_sequences)}, 测试集: {len(test_sequences)}")

if __name__ == "__main__":
    main()
