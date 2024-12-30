import math, os, json, torch, datetime, random, copy, shutil, torchvision, time
# import sys
# sys.path.append('../..')
import torch.nn as nn
import torch.optim as Optim
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
from collections import namedtuple

# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP

# from mmnas.loader.load_data_vqa import DataSet
# from mmnas.loader.filepath_vqa import Path
# from mmnas.utils.vqa import VQA
# from mmnas.utils.vqaEval import VQAEval
from model.full_model_AttDetail import Net_Full
from utils.optimizer import WarmupOptimizer
# from mmnas.utils.sampler import SubsetDistributedSampler
from loader.path import Path
import utils.eval as eval
import time
from utils.architect import Architect
from utils.log import log_training_data, combine_genotype_probs
from loader.dataset_per_label import load_retrain_data, load_test_data     # 修改
from loss.loss import InfoNCE
from tensorboardX import SummaryWriter

RUN_MODE = 'train'
# RUN_MODE = 'val'
# RUN_MODE = 'test'


class CfgSearch():
    def __init__(self):
        super(CfgSearch, self).__init__()

        # search
        self.DATASET_CONFIG = './json/Anet.json'              # 参数 0
        self.CKPT_PATH = '/mnt/add_disk/NAS/ActNet/Log_24/'     # 参数 1
        # self.ARCH_PATH = self.CKPT_PATH + 'arch.json'
        # self.PROB_PATH = self.CKPT_PATH + 'prob.json' # 选中的操作的 prob
        # self.PROBS_PATH = self.CKPT_PATH + 'probs.json' # 所有操作的 probs
        self.BEST_ARCH_PATH_ONE = self.CKPT_PATH + 'best_arch_one.json' # 记录最好的一
        self.BEST_PROB_PATH_ONE = self.CKPT_PATH + 'best_prob_one.json' # 记录最好的一
        self.BEST_ARCH_PATH_TWO = self.CKPT_PATH + 'best_arch_two.json' # 记录最好的二
        self.BEST_PROB_PATH_TWO = self.CKPT_PATH + 'best_prob_two.json' # 记录最好的二
        self.BEST_PROBS_PATH = self.CKPT_PATH + 'best_probs.json'
        self.WEIGHT_PATH = self.CKPT_PATH + 'weigth.pth'
        self.LOG_PATH = self.CKPT_PATH + 'log.txt'

        self.DEVICE = 'cuda:2'  # 参数 3
        self.LR = 0.0001        # 参数 4

        self.HASHCODE_SIZE = 128 # 参数 6
        self.MAX_ITER = 300     # 参数 7
        self.RESUME_WEIGHT_PATH = None

        # retrain dir
        self.ONE_TWO = 'one'                                # 参数 2
        self.retrainWithAlpha = False                        # 参数 5
        self.TIMESTAMP = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.RETRAIN_DIR = self.CKPT_PATH + '/retrain/' + str(self.TIMESTAMP)
        if not os.path.exists(self.RETRAIN_DIR):
            # 如果目录不存在，则创建它
            os.makedirs(self.RETRAIN_DIR)

        # retrain
        if self.ONE_TWO == 'one':
            self.GENOTYPE = json.load(open(self.BEST_ARCH_PATH_ONE, 'r+'))
        elif self.ONE_TWO == 'two':
            self.GENOTYPE = json.load(open(self.BEST_ARCH_PATH_TWO, 'r+'))
        self.TENSORBOARD_PATH  = self.CKPT_PATH + '/runs/' + str(self.TIMESTAMP)
        self.RETRAIN_WEIGHT_PATH = self.RETRAIN_DIR + '/retrain_weigth.pth'
        self.RETRAIN_LOG_PATH = self.RETRAIN_DIR + '/retrain_log.txt'
        self.RETRAIN_LOG_PROBS = self.RETRAIN_DIR + '/retrain_probs.txt'

        # Set Seed For CPU And GPUs
        self.SEED = 3346
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)
        torch.cuda.manual_seed_all(self.SEED)
        np.random.seed(self.SEED)
        random.seed(self.SEED)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


        # data loader
        self.NUM_WORKERS = 4
        self.BATCH_SIZE = 128
        self.EVAL_BATCH_SIZE = self.BATCH_SIZE

        # Network Params
        self.HSIZE = 768        # 参数 8
        self.DROPOUT_R = 0.1
        self.OPS_RESIDUAL = False
        self.OPS_NORM = False
        # Mamba Params
        self.d_model = self.HSIZE
        self.dt_rank = 4
        self.d_state: int = 16
        self.expand_factor: int = 2 # 参数 9
        self.d_conv: int = 4
        self.d_inner = self.expand_factor * self.d_model
        self.dt_min: float = 0.001
        self.dt_max: float = 0.1
        self.dt_init: str = "random"  #  "random" or "constant"
        self.dt_scale: float = 1.0
        self.dt_init_floor = 1e-4
        self.conv_bias: bool = True
        self.modal_bias: float = 0.8
        self.bias: bool = False
        self.pscan: bool = True  #  use parallel scan mode or sequential mode when training
        # Transformer Params
        self.attention_probs_dropout_prob = 0.1
        self.layer_norm_eps = 1e-12
        self.hidden_dropout_prob = 0.1
        self.intermediate_size = 3072

        # gene_key
        self.ImgEnc = "ImgEnc"
        self.AudEnc = "AudEnc"
        self.Inter = "Inter"
        self.Fusion = "Fusion"
        self.Final = "Final"

        for key, value in vars(self).items():
            log_training_data(f'{key}: {value}', self.RETRAIN_LOG_PATH)

        #
        self.writer = SummaryWriter(self.TENSORBOARD_PATH)

class Execution:
    def __init__(self, __C):
        self.__C = __C

    def get_optim(self, net):

        net_optim = torch.optim.Adam(net.parameters(), lr=self.__C.LR, eps=1e-3)

        return net_optim

    def train(self, train_loader, model, criterion, net_optim, epoch):

        model.train()
        # model
        model_loss = 0.

        time_start = time.time()
        for index, data in enumerate(train_loader):
            if index % 50 == 0:
                print(f'model {epoch} {index}')

            anchorI, anchorA, posI, posA, negI, negA = data

            anchorI = anchorI.to(self.__C.DEVICE)
            anchorA = anchorA.to(self.__C.DEVICE)
            posI = posI.to(self.__C.DEVICE)
            posA = posA.to(self.__C.DEVICE)
            negI = negI.to(self.__C.DEVICE)
            negA = negA.to(self.__C.DEVICE)

            batch, negs = negI.shape[0], negI.shape[1]
            negI_ = negI.view(-1, 25, 768)
            negA_ = negA.view(-1, 25, 768)

            anchor_vh = model(anchorI, anchorA)
            pos_vh = model(posI, posA)
            neg_vh = model(negI_, negA_)
            neg_vh = neg_vh.view(batch, negs, -1)

            # 转换形状
            lv = criterion(anchor_vh, pos_vh, neg_vh)

            net_optim.zero_grad()
            lv.backward()
            net_optim.step()
            model_loss += lv.item()

        time_end = time.time() - time_start

        epoch_model_loss = model_loss / len(train_loader)

        split = '-' * 20
        epo = 'Epoch {}/{}'.format(epoch, self.__C.MAX_ITER)
        ml = 'model Loss: {:.6f}'.format(epoch_model_loss)
        t  = 'Train time {:.0f}m {:.0f}s'.format(time_end // 60, time_end % 60)
        info = split + '\n' + epo + '\n' + ml + '\n' + t

        log_training_data(info, self.__C.RETRAIN_LOG_PATH)

        self.__C.writer.add_scalar('retrain_model', epoch_model_loss, global_step=epoch)

    def retrain(self, train_loader, valid_loader, data_loader):

        model = Net_Full(self.__C)
        for n, p in model.named_parameters():
            log_training_data(n,self.__C.RETRAIN_LOG_PATH)
        # print(model)
        model.to(self.__C.DEVICE)

        if self.__C.RESUME_WEIGHT_PATH:
            state_dict = torch.load(self.__C.WEIGHT_PATH, map_location=self.__C.DEVICE)
            model.load_state_dict(state_dict['berthash'], strict=True)
            print('Finish loading ckpt !!!')

        criterion = InfoNCE(negative_mode='paired')

        net_optim = self.get_optim(model)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     net_optim, self.__C.MAX_ITER, eta_min=self.__C.NET_LR_MIN)

        best_mAP = 0.
        for epoch in range(0, self.__C.MAX_ITER):


            # training
            self.train(train_loader, model, criterion, net_optim, epoch)


            # validation
            mAP = self.validate(valid_loader, data_loader, model)

            info = 'mAP: {:.4f}'.format(mAP)
            log_training_data(info, self.__C.RETRAIN_LOG_PATH)
            self.__C.writer.add_scalar('mAP', mAP, global_step=epoch)



            # save
            if best_mAP < mAP:
                best_mAP = mAP
                state = {
                    'state_dict': model.state_dict(),
                    'net_optim': net_optim.state_dict(),
                }

                torch.save(state, self.__C.RETRAIN_WEIGHT_PATH)

            if self.__C.retrainWithAlpha:
                probs = model.genotype_weights()
                if epoch == 0:
                    json.dump({}, open(self.__C.RETRAIN_LOG_PROBS, 'w+'))
                formatted_arch_json = json.dumps(probs, indent=4)
                print(formatted_arch_json)
                probs_json = json.load(open(self.__C.RETRAIN_LOG_PROBS, 'r+'))
                probs_json['epoch' + str(epoch)] = probs
                json.dump(probs_json, open(self.__C.RETRAIN_LOG_PROBS, 'w+'), indent=4)



    def validate(self, valid_loader, data_loader, model):

        model.eval()
        with torch.no_grad():
            q_image_code, q_image_targets = self.generate_code(model, valid_loader, self.__C.HASHCODE_SIZE,
                                                               self.__C.DEVICE)
            r_image_code, r_image_targets = self.generate_code(model, data_loader, self.__C.HASHCODE_SIZE,
                                                               self.__C.DEVICE)

            mAP = eval.mean_average_precision(
                q_image_code.to(self.__C.DEVICE),
                r_image_code.to(self.__C.DEVICE),
                q_image_targets.to(self.__C.DEVICE),
                r_image_targets.to(self.__C.DEVICE),
                self.__C.DEVICE
            )

        return mAP


    def generate_code(self, model, dataloader, code_length, device):
        model.eval()
        with torch.no_grad():
            N = len(dataloader.dataset)
            numclass = dataloader.dataset.num_classes
            code = torch.zeros([N, code_length])
            target = torch.zeros([N, numclass])
            for image_features, audio_features, tar, index in dataloader:
                # for data, data_mask, tar, index in dataloader:
                image_features = image_features.to(device)
                audio_features = audio_features.to(device)

                hash_code = model(image_features, audio_features)
                code[index, :] = hash_code.sign().cpu()
                target[index, :] = tar.clone().cpu()
        torch.cuda.empty_cache()
        return code, target


    def run(self):
        if RUN_MODE in ['train']:

            # dataloader
            train_loader, valid_loader, data_loader = load_retrain_data(
                data_set_config=self.__C.DATASET_CONFIG
                , batch_size=self.__C.BATCH_SIZE
                , num_workers=self.__C.NUM_WORKERS
                , pn=5
            )
            print(len(train_loader))  # 157
            print(len(valid_loader)) # 40
            print(len(data_loader)) # 157

            self.retrain(train_loader, valid_loader, data_loader)

        elif RUN_MODE in ['val', 'test']:

            model = Net_Full(self.__C)
            model.to(self.__C.DEVICE)

            if self.__C.RESUME_WEIGHT_PATH:
                state_dict = torch.load(self.__C.WEIGHT_PATH, map_location=self.__C.DEVICE)
                model.load_state_dict(state_dict['berthash'], strict=True)
                print('Finish loading ckpt !!!')

            test_loader, data_loader = load_test_data(
                data_set_config=self.__C.DATASET_CONFIG
                , batch_size=self.__C.BATCH_SIZE
                , num_workers=self.__C.NUM_WORKERS
            )

            self.validate(test_loader, data_loader, model)

        else:
            exit(-1)

if __name__ == '__main__':
    __C = CfgSearch()
    exec = Execution(__C)
    exec.run()
    # model = Net_Full(__C)
    # print(model)