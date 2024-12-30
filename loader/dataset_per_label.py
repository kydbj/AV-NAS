import random
import json
import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

class TrainDataset(Dataset):  #  video_name, frame_count, class_label = line.strip().split(',')[0:3]  这一行要修改一下

    def __init__(self, config_file, image_features, audio_features, pn, phase=""):
        with open(config_file, 'r') as f:
            config = json.load(f)
            f.close()
        self.dataset = config['dataset']
        self.image_features = image_features # 从h5文件中读取出来的特征,key-value形式
        self.audio_features = audio_features # 从h5文件中读取出来的特征,key-value形式

        if phase == 'train':
            self.list_file = config['train_list']
        elif phase == 'valid':
            self.list_file = config['val_list']

        self.pn = pn

        self.data = self.load_videos_from_file(self.list_file) # video_data[1] = [a,b,c,d,...]: 属于第一类的视频名字有a,b,c,d...
        self.classes = list(self.data.keys()) # [1,2,3,4..]
        self.indexes = self._make_indexes(self.data) #  indexes: [(0,1),(1,1),(2,1),(3,1),...(0,2),(1,2),(2,2),(3,2)]


    # video_data[1] = [a,b,c,d,...] : 属于第一类的视频名字有a,b,c,d...
    # video_data[2] = [e,f,g,h,...] : 属于第一类的视频名字有e,f,g,h,...
    def load_videos_from_file(self,file_path):
        video_data = {}
        if self.dataset == 'actnet':
            with open(file_path, 'r') as file:
                for line in file:
                    video_name, frame_count, class_label = line.strip().split(',')[0:3]
                    class_label = int(class_label)  # Convert class label to integer

                    if class_label not in video_data:
                        video_data[class_label] = []

                    video_data[class_label].append(video_name)
        elif self.dataset == 'fcvid':
            with open(file_path, 'r') as file:
                for line in file:
                    video_name, class_label = line.strip().split(',')[0:2]
                    class_label = int(class_label)  # Convert class label to integer

                    if class_label not in video_data:
                        video_data[class_label] = []

                    # Append the video information to the class list
                    video_data[class_label].append(video_name)

        return video_data

    #   video_data[1] = [a,b,c,d,...]； video_data[2] = [a,b,c,d,...]；
    #   indexes: [(0,1),(1,1),(2,1),(3,1),...(0,2),(1,2),(2,2),(3,2)]
    def _make_indexes(self, video_data):
        """
        Create a list of indexes for all videos, maintaining class information.
        :param video_data: A dictionary with class labels and their corresponding videos.
        :return: A list of tuples, each containing the video index and its class label.
        """
        indexes = []
        for class_label, videos in video_data.items():
            for i in range(len(videos)):
                indexes.append((i, class_label))
        return indexes


    # def select_n_elements_from_k_keys(self, divtKeys, K, N):
    #     selected_keys = random.sample(divtKeys, K)
    #     all_elements = []
    #     for key in selected_keys:
    #         all_elements.extend(self.data[key])
    #     selected_elements = random.sample(all_elements, N)
    #     return selected_elements
    def select_n_neg_elements(self, divtKeys, N):
        selected_keys = random.sample(divtKeys, N)
        all_elements = []
        for key in selected_keys:
            all_elements.extend(random.sample(self.data[key],1))
        return all_elements


    # index： indexes: [(0,1),(1,1),(2,1),(3,1),...(0,2),(1,2),(2,2),(3,2)]中的下标，整体下标
    def __getitem__(self, index):


        # anchor
        class_index, class_label = self.indexes[index]
        anchor = self.data[class_label][class_index] # video 的名字
        # positive
        pos_class_index = (class_index + random.randint(1, len(self.data[class_label]) - 1)) % len(
                self.data[class_label])
        pos = self.data[class_label][pos_class_index]
        # negtive
        neg_num = self.pn - 1
        neg_classes = [cls for cls in self.classes if cls != class_label]
        negs = self.select_n_neg_elements(neg_classes, neg_num)

        # 返回特征
        anchorI, anchorA = self.image_features[anchor], self.audio_features[anchor]
        posI, posA = self.image_features[pos], self.audio_features[pos]
        negI = []
        negA = []
        for neg in negs:
            negI.append(self.image_features[neg])
            negA.append(self.audio_features[neg])

        return anchorI, anchorA, posI, posA, torch.stack(negI,dim=0), torch.stack(negA,dim=0)

    def __len__(self):
        """
        Get the total number of videos.
        :return: Number of videos.
        """
        return len(self.indexes)


class QRDataset(Dataset):
    def __init__(self, config_file, image_features, audio_features, phase='retrieval'): # val, test, dataset
        with open(config_file, 'r') as f:
            config = json.load(f)


        self.dataset = config['dataset']
        # 数据名称
        if phase == 'valid':
            self.list_file = config['val_list']
        elif phase == 'test':
            self.list_file = config['test_list']
        elif phase == 'retrieval':
            self.list_file = config['retrieval_list']
        self.num_classes = config['num_class']
        self.image_features = image_features
        self.audio_features = audio_features


        self.video_data = self.load_videos_from_file(self.list_file) # 名称加载成字典
        self.indexes = self._make_indexes(self.video_data) # (类别，名称)


    # 将.txt文件加载成字典
    def load_videos_from_file(self,file_path):
        video_data = {}
        if self.dataset == 'actnet':
            with open(file_path, 'r') as file:
                for line in file:
                    video_name, frame_count, class_label = line.strip().split(',')[0:3]
                    class_label = int(class_label)  # Convert class label to integer

                    if class_label not in video_data:
                        video_data[class_label] = []

                    # Append the video information to the class list
                    video_data[class_label].append(video_name)
        elif self.dataset == 'fcvid':
            with open(file_path, 'r') as file:
                for line in file:
                    video_name, class_label = line.strip().split(',')[0:2]
                    class_label = int(class_label)  # Convert class label to integer

                    if class_label not in video_data:
                        video_data[class_label] = []

                    # Append the video information to the class list
                    video_data[class_label].append(video_name)

        return video_data

    def _make_indexes(self, video_data):
        """
        Create a list of indexes for all videos, maintaining class information.
        :param video_data: A dictionary with class labels and their corresponding videos.
        :return: A list of tuples, each containing the video index and its class label.
        """
        indexes = []
        for class_label, videos in video_data.items():
            for i in range(len(videos)):
                indexes.append((i, class_label))
        print('video number:%d' % (len(indexes)))
        return indexes


    def __getitem__(self, index):
        video_index, class_label = self.indexes[index] #
        video = self.video_data[class_label][video_index] # video 是一个视频名字

        image_features = self.image_features[video]
        audio_features = self.audio_features[video]

        if self.dataset == 'actnet':
            target_onehot = torch.nn.functional.one_hot(torch.tensor(class_label), num_classes=self.num_classes).float() # 标签独热码
        elif self.dataset == 'fcvid':
            target_onehot = torch.nn.functional.one_hot(torch.tensor(class_label-1), num_classes=self.num_classes).float() # 标签独热码

        return image_features,audio_features,target_onehot, index

    def __len__(self):
        return len(self.indexes)



def load_h5_file_to_memory(image_features_path,audio_faetures_path):
    image_features = {}
    audio_features = {}
    with h5py.File(image_features_path, 'r') as h5_file1:
        for key in h5_file1.keys():
            image_features[key] = torch.tensor(h5_file1[key]['vectors'][...])


    with h5py.File(audio_faetures_path, 'r') as h5_file2:
        for key in h5_file2.keys():
            audio_features[key] = torch.tensor(h5_file2[key]['vectors'][...])
    return image_features, audio_features


def load_test_data(data_set_config,batch_size,num_workers):


    with open(data_set_config, 'r') as f:
        config = json.load(f)
        f.close()
    image_features, audio_features = load_h5_file_to_memory(config['Image'],config['Audio'])

    query_dataloader = DataLoader(
        QRDataset(config_file=data_set_config,
                  image_features=image_features,
                  audio_features=audio_features,
                  phase='test',
                 ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    retrival_dataloader = DataLoader(
        QRDataset(config_file=data_set_config,
                  image_features=image_features,
                  audio_features=audio_features,
                  phase='retrieval',
                  ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )


    return  query_dataloader, retrival_dataloader

def load_data(data_set_config,batch_size,num_workers,pn):


    with open(data_set_config, 'r') as f:
        config = json.load(f)
        f.close()
    image_fratures,audio_features = load_h5_file_to_memory(config['Image'],config['Audio'])

    train_dataloader = DataLoader(
        TrainDataset(config_file=data_set_config,   # 输出 特征和对应标签
                     image_features=image_fratures,
                     audio_features=audio_features,
                     pn=pn,
                     phase="train"
                     ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    valid_dataloader = DataLoader(
        TrainDataset(config_file=data_set_config,   # 输出 特征和对应标签
                     image_features=image_fratures,
                     audio_features=audio_features,
                     pn=pn,
                     phase="valid"
                     ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        QRDataset(config_file=data_set_config,
                  image_features=image_fratures,
                  audio_features=audio_features,
                  phase='test',
                 ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    retrival_dataloader = DataLoader(
        QRDataset(config_file=data_set_config,
                  image_features=image_fratures,
                  audio_features=audio_features,
                  phase='retrieval',
                  ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )
    return train_dataloader, valid_dataloader, test_dataloader, retrival_dataloader


def load_darts_data(data_set_config,batch_size,num_workers,pn):


    with open(data_set_config, 'r') as f:
        config = json.load(f)
        f.close()
    image_fratures,audio_features = load_h5_file_to_memory(config['Image'],config['Audio'])

    train_data = TrainDataset(config_file=data_set_config,  # 输出 特征和对应标签
                 image_features=image_fratures,
                 audio_features=audio_features,
                 pn=pn,
                 phase="train"
                 )

    n_train = len(train_data)
    split = n_train // 2
    indices = list(range(n_train))

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    valid_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=valid_sampler,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        QRDataset(config_file=data_set_config,
                  image_features=image_fratures,
                  audio_features=audio_features,
                  phase='valid',
                 ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    retrival_dataloader = DataLoader(
        QRDataset(config_file=data_set_config,
                  image_features=image_fratures,
                  audio_features=audio_features,
                  phase='retrieval',
                  ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )
    return train_dataloader, valid_dataloader, test_dataloader, retrival_dataloader

def load_retrain_data(data_set_config,batch_size,num_workers,pn):


    with open(data_set_config, 'r') as f:
        config = json.load(f)
        f.close()
    image_fratures,audio_features = load_h5_file_to_memory(config['Image'],config['Audio'])

    train_dataloader = DataLoader(
        TrainDataset(config_file=data_set_config,   # 输出 特征和对应标签
                     image_features=image_fratures,
                     audio_features=audio_features,
                     pn=pn,
                     phase="train"
                     ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    valid_dataloader = DataLoader(
        QRDataset(config_file=data_set_config,
                  image_features=image_fratures,
                  audio_features=audio_features,
                  phase='valid',
                 ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    retrival_dataloader = DataLoader(
        QRDataset(config_file=data_set_config,
                  image_features=image_fratures,
                  audio_features=audio_features,
                  phase='retrieval',
                  ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )
    return train_dataloader, valid_dataloader, retrival_dataloader

if __name__ == '__main__':


    A,B,C,D = load_darts_data(data_set_config='../json/Anet.json',batch_size=2,num_workers=2,pn=5)

    print(len(A))
    print(len(B))

    print(A)
    print(B)


    for ind, (O1, O2) in enumerate(zip(A,B)):
        O1 = [v.to("cuda:0") for v in O1]
        O2 = [v.to("cuda:0") for v in O2]

        trn_anchorI, trn_anchorA, trn_posI, trn_posA, trn_negI, trn_negA = O1
        val_anchorI, val_anchorA, val_posI, val_posA, val_negI, val_negA = O2

        print(trn_negA.shape)
        print(val_negA.shape)
