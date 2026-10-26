import os
import scipy.io as scio
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def decode_config(config):   # 根据 config.yaml 解析数据划分（train/val）
    all_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14',
                    'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28',
                    'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
    all_actions = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14',
                   'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27']
    train_form = {}   # 训练集中每个被试对应的动作集合
    val_form = {}   # 验证集中每个被试对应的动作集合
    # Limitation to actions (protocol)
    if config['protocol'] == 'protocol1':  # Daily actions
        actions = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A27']
    elif config['protocol'] == 'protocol2':  # Rehabilitation actions:
        actions = ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A15', 'A16', 'A24', 'A25', 'A26']
    else:
        actions = all_actions

    if config['split_to_use'] == "kfold_cross_subject":  # K-Fold Cross-Subject Validation
        kfold_cfg = config["kfold"]  # 获取 K-Fold 配置
        subjects_test = kfold_cfg["subjects_test"]  # 测试集被试列表
        subjects_trainval = kfold_cfg["subjects_trainval"]  # 训练验证集被试列表  ***具体看config.yaml***

        num_folds = kfold_cfg["num_folds"]  # 折数
        current_fold = kfold_cfg["current_fold"]  # 当前折数

        assert num_folds > 1, "num_folds must be >= 2"  # 折数必须大于 1
        assert 0 <= current_fold < num_folds, "current_fold out of range"  # 当前折数必须在范围内

        # Compute fold partition
        fold_size = len(subjects_trainval) // num_folds  # 每折的大小
        val_start = current_fold * fold_size  # 验证集起始索引
        val_end = (current_fold + 1) * fold_size if current_fold != num_folds - 1 else len(subjects_trainval)  # 验证集结束索引

        subjects_val = subjects_trainval[val_start:val_end]  # 验证集被试列表
        subjects_train = [s for s in subjects_trainval if s not in subjects_val]  # 训练集被试列表

        print(f"[KFold mode] Fold {current_fold+1}/{num_folds}")
        print("Train subjects:", subjects_train)
        print("Val subjects:", subjects_val)
        print("Test subjects:", subjects_test)

        # actions = config.get("actions", "all")

        train_form = {s: actions for s in subjects_train}  # 训练集数据索引
        val_form = {s: actions for s in subjects_val}  # 验证集数据索引
        test_form = {s: actions for s in subjects_test}  # 测试集数据索引
        # return train_form, val_form, test_form
        dataset_config = {
        "train_dataset": {"modality": config["modality"], "split": "training", "data_form": train_form},
        "val_dataset": {"modality": config["modality"], "split": "validation", "data_form": val_form},
        "test_dataset": {"modality": config["modality"], "split": "testing", "data_form": test_form},
        }
        return dataset_config

class MMFi_Database:   # 扫描整个数据目录，建立树状索引结构
    def __init__(self, data_root):
        self.data_root = data_root
        self.scenes = {}
        self.subjects = {}
        self.actions = {}
        self.modalities = {}     # 为每种访问方式建立索引，后续可以根据「场景、被试、动作、模态」四种方式任意访问数据路径。
        self.load_database()

    def load_database(self):
        for scene in sorted(os.listdir(self.data_root)):
            if scene.startswith("."):
                continue
            self.scenes[scene] = {}
            for subject in sorted(os.listdir(os.path.join(self.data_root, scene))):
                if subject.startswith("."):
                    continue
                self.scenes[scene][subject] = {}
                self.subjects[subject] = {}
                for action in sorted(os.listdir(os.path.join(self.data_root, scene, subject))):
                    if action.startswith("."):
                        continue
                    self.scenes[scene][subject][action] = {}
                    self.subjects[subject][action] = {}
                    if action not in self.actions.keys():
                        self.actions[action] = {}
                    if scene not in self.actions[action].keys():
                        self.actions[action][scene] = {}
                    if subject not in self.actions[action][scene].keys():
                        self.actions[action][scene][subject] = {}
                    for modality in ['infra1', 'infra2', 'depth', 'rgb', 'lidar', 'mmwave', 'wifi-csi']:
                        data_path = os.path.join(self.data_root, scene, subject, action, modality)
                        self.scenes[scene][subject][action][modality] = data_path
                        self.subjects[subject][action][modality] = data_path
                        self.actions[action][scene][subject][modality] = data_path
                        if modality not in self.modalities.keys():
                            self.modalities[modality] = {}
                        if scene not in self.modalities[modality].keys():
                            self.modalities[modality][scene] = {}
                        if subject not in self.modalities[modality][scene].keys():
                            self.modalities[modality][scene][subject] = {}
                        if action not in self.modalities[modality][scene][subject].keys():
                            self.modalities[modality][scene][subject][action] = data_path


class MMFi_Dataset(Dataset):   # 真正加载数据内容
    def __init__(self, data_base, data_unit, modality, split, data_form):
        self.data_base = data_base
        self.data_unit = data_unit
        self.modality = modality.split('|')   # 支持多模态输入，以 '|' 分隔
        for m in self.modality:
            assert m in ['rgb', 'infra1', 'infra2', 'depth', 'lidar', 'mmwave', 'wifi-csi']
        self.split = split
        self.data_source = data_form
        self.data_list = self.load_data()

    def get_scene(self, subject):
        if subject in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']:
            return 'E01'
        elif subject in ['S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']:
            return 'E02'
        elif subject in ['S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']:
            return 'E03'
        elif subject in ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']:
            return 'E04'
        else:
            raise ValueError('Subject does not exist in this dataset.')

    def get_data_type(self, mod):
        if mod in ["rgb", 'infra1', "infra2"]:
            return ".npy"
        elif mod in ["lidar", "mmwave"]:
            return ".bin"
        elif mod in ["depth"]:
            return ".png"
        elif mod in ["wifi-csi"]:
            return ".mat"
        else:
            raise ValueError("Unsupported modality.")

    def load_data(self):   # 生成每一条样本的路径信息
        data_info = []
        for subject, actions in self.data_source.items():
            for action in actions:
                if self.data_unit == 'sequence':   # 整段动作作为一个样本
                    data_dict = {'modality': self.modality,
                                 'scene': self.get_scene(subject),
                                 'subject': subject,
                                 'action': action,
                                 'gt_path': os.path.join(self.data_base.data_root, self.get_scene(subject), subject,
                                                         action, 'ground_truth.npy')
                                 }
                    for mod in self.modality:
                        data_dict[mod+'_path'] = os.path.join(self.data_base.data_root, self.get_scene(subject), subject,
                                                         action, mod)
                    data_info.append(data_dict)
                elif self.data_unit == 'frame':    # 每一帧作为一个样本
                    frame_num = 297
                    for idx in range(frame_num):
                        data_dict = {'modality': self.modality,
                                     'scene': self.get_scene(subject),
                                     'subject': subject,
                                     'action': action,
                                     'gt_path': os.path.join(self.data_base.data_root, self.get_scene(subject), subject,
                                                             action, 'ground_truth.npy'),
                                     'idx': idx
                                     }
                        data_valid = True
                        for mod in self.modality:
                            data_dict[mod+'_path'] = os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, mod, "frame{:03d}".format(idx+1) + self.get_data_type(mod))

                            if os.path.getsize(data_dict[mod+'_path']) == 0:
                                data_valid = False
                        if data_valid:
                            data_info.append(data_dict)
                else:
                    raise ValueError('Unsupport data unit!')
        return data_info

    def read_dir(self, dir):   # 根据模态自动读取文件格式
        _, mod = os.path.split(dir)
        data = []
        if mod in ['infra1', 'infra2', 'rgb']:  # rgb, infra1 and infra2 are 2D keypoints   2D关键点特征
            for arr_file in sorted(glob.glob(os.path.join(dir, "frame*.npy"))):
                arr = np.load(arr_file)
                data.append(arr)
            data = np.array(data)
        elif mod == 'depth':   # 深度图
            for img in sorted(glob.glob(os.path.join(dir, "frame*.png"))):
                _cv_img = cv2.imread(img, cv2.IMREAD_UNCHANGED)  # Default depth value is 16-bit
                _cv_img *= 0.001  # Convert unit to meter
                data.append(_cv_img)
            data = np.array(data)
        elif mod == 'lidar':   # 3D点云，3列
            for bin_file in sorted(glob.glob(os.path.join(dir, "frame*.bin"))):
                with open(bin_file, 'rb') as f:
                    raw_data = f.read()
                    data_tmp = np.frombuffer(raw_data, dtype=np.float64)
                    data_tmp = data_tmp.reshape(-1, 3)
                data.append(data_tmp)
        elif mod == 'mmwave':   # 毫米波点云，5列：x, y, z, v, p
            for bin_file in sorted(glob.glob(os.path.join(dir, "frame*.bin"))):
                with open(bin_file, 'rb') as f:
                    raw_data = f.read()
                    data_tmp = np.frombuffer(raw_data, dtype=np.float64)
                    data_tmp = data_tmp.copy().reshape(-1, 5)
                    # data_tmp = data_tmp[:, :3]
                data.append(data_tmp)
        elif mod == 'wifi-csi':   # CSI幅值矩阵
            for csi_mat in sorted(glob.glob(os.path.join(dir, "frame*.mat"))):
                data_mat = scio.loadmat(csi_mat)['CSIamp']
                data_mat[np.isinf(data_mat)] = np.nan
                for i in range(10):  # 32
                    temp_col = data_mat[:, :, i]
                    nan_num = np.count_nonzero(temp_col != temp_col)
                    if nan_num != 0:
                        temp_not_nan_col = temp_col[temp_col == temp_col]
                        temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
                data_mat = (data_mat - np.min(data_mat)) / (np.max(data_mat) - np.min(data_mat))
                data_frame = np.array(data_mat)
                data.append(data_frame)
            data = np.array(data)
        else:
            raise ValueError('Found unseen modality in this dataset.')
        return data

    def read_frame(self, frame):
        _mod, _frame = os.path.split(frame)
        _, mod = os.path.split(_mod)
        if mod in ['infra1', 'infra2', 'rgb']:
            data = np.load(frame)
        elif mod == 'depth':
            data = cv2.imread(frame, cv2.IMREAD_UNCHANGED) * 0.001
        elif mod == 'lidar':
            with open(frame, 'rb') as f:
                raw_data = f.read()
                data = np.frombuffer(raw_data, dtype=np.float64)
                data = data.reshape(-1, 3)
        elif mod == 'mmwave':
            with open(frame, 'rb') as f:
                raw_data = f.read()
                data = np.frombuffer(raw_data, dtype=np.float64)
                data = data.copy().reshape(-1, 5)
                # data = data[:, :3]
        elif mod == 'wifi-csi':
            data = scio.loadmat(frame)['CSIamp']
            data[np.isinf(data)] = np.nan
            for i in range(10):  # 32
                temp_col = data[:, :, i]
                nan_num = np.count_nonzero(temp_col != temp_col)
                if nan_num != 0:
                    temp_not_nan_col = temp_col[temp_col == temp_col]
                    temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
        else:
            raise ValueError('Found unseen modality in this dataset.')
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):   # 根据索引读取数据内容
        item = self.data_list[idx]

        gt_numpy = np.load(item['gt_path'])
        gt_torch = torch.from_numpy(gt_numpy)

        if self.data_unit == 'sequence':
            sample = {'modality': item['modality'],
                      'scene': item['scene'],
                      'subject': item['subject'],
                      'action': item['action'],
                      'output': gt_torch
                      }
            for mod in item['modality']:
                data_path = item[mod+'_path']
                if os.path.isdir(data_path):
                    data_mod = self.read_dir(data_path)
                else:
                    data_mod = np.load(data_path + '.npy')
                sample['input_'+mod] = data_mod
        elif self.data_unit == 'frame':
            sample = {'modality': item['modality'],
                      'scene': item['scene'],
                      'subject': item['subject'],
                      'action': item['action'],
                      'idx': item['idx'],
                      'output': gt_torch[item['idx']]
                      }
            for mod in item['modality']:
                data_path = item[mod + '_path']
                if os.path.isfile(data_path):
                    data_mod = self.read_frame(data_path)
                    sample['input_'+mod] = data_mod
                else:
                    raise ValueError('{} is not a file!'.format(data_path))
        else:
            raise ValueError('Unsupport data unit!')
        return sample

def make_dataset(dataset_root, config):
    database = MMFi_Database(dataset_root)  # 创建数据库
    config_dataset = decode_config(config)  # 解析数据划分
    train_dataset = MMFi_Dataset(database, config['data_unit'], **config_dataset['train_dataset'])
    val_dataset = MMFi_Dataset(database, config['data_unit'], **config_dataset['val_dataset'])
    test_dataset = MMFi_Dataset(database, config['data_unit'], **config_dataset['test_dataset'])
    return train_dataset, val_dataset, test_dataset

def collate_fn_padd(batch, max_points=112): # # 点云最大点数设为 112
    """
    仅对每帧点云进行二级 padding（点数补齐到 max_points）
    适配固定帧数（297帧）+ 稀疏点云输入的 MMFi 数据。
    """

    batch_data = {
        'modality': batch[0]['modality'],  # 模态
        'scene': [sample['scene'] for sample in batch],  # 场景ID
        'subject': [sample['subject'] for sample in batch],  # 人员ID
        'action': [sample['action'] for sample in batch],  # 动作名称
        'idx': [sample['idx'] for sample in batch] if 'idx' in batch[0] else None  # 帧索引
    }

    # 标签堆叠
    outputs = [np.array(sample['output']) for sample in batch]  # 输出标签
    batch_data['output'] = torch.FloatTensor(np.array(outputs))  # 输出标签

    for mod in batch_data['modality']:  # 分模态处理输入数据
        if mod in ['mmwave', 'lidar']:  # 点云数据需要二级 padding
            # 每个样本: shape = (T=297, Ni, 5)
            padded_batch = []  # 存放每个样本的 padded 结果

            for sample in batch:
                seq = sample['input_' + mod]  # 每个样本 (297, Ni, 5)
                padded_seq = []  # 存放每一帧的 padded 结果
                for frame in seq:  # 遍历每一帧
                    num_points = frame.shape[0]  # 当前帧点数

                    if num_points == 0:  # 无点 → 全零
                        pad_frame = np.zeros((max_points, frame.shape[1]), dtype=np.float32)  # (max_points, 5)
                    elif num_points >= max_points:  # 超过上限
                        # 超过上限 → 随机采样 max_points 个点
                        idx = np.random.choice(num_points, max_points, replace=False) 
                        pad_frame = frame[idx] 
                    else:
                        # 不足 → 补零
                        pad_frame = np.zeros((max_points, frame.shape[1]), dtype=np.float32) 
                        pad_frame[:num_points, :] = frame

                    padded_seq.append(pad_frame)

                # 拼成 (T, max_points, 5)
                padded_seq = np.stack(padded_seq, axis=0)
                padded_batch.append(padded_seq)

            # 拼成 (B, T, max_points, 5)
            batch_data['input_' + mod] = torch.FloatTensor(np.array(padded_batch))

        else:
            # 其他模态（RGB/depth等）直接堆叠
            inputs = [np.array(sample['input_' + mod]) for sample in batch]
            batch_data['input_' + mod] = torch.FloatTensor(np.array(inputs))

    return batch_data

def make_dataloader(dataset, is_training, generator, batch_size, collate_fn_padd = collate_fn_padd):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_padd,
        shuffle=is_training,
        drop_last=is_training,
        generator=generator
    )
    return loader
