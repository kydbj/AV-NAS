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


from model.hygr_model_AttDetail import Net_Search
from utils.optimizer import WarmupOptimizer
# from mmnas.utils.sampler import SubsetDistributedSampler
from loader.path import Path
import utils.eval as eval
from utils.architect import Architect
from utils.log import log_training_data, combine_genotype_probs
from loader.dataset_per_label import load_retrain_data     # 修改
from loss.loss import InfoNCE
from tensorboardX import SummaryWriter

RUN_MODE = 'train'
# RUN_MODE = 'val'
# RUN_MODE = 'test'

# 自己优化
class CfgSearch(Path):
    def __init__(self):
        super(CfgSearch, self).__init__()

        self.WORKSPACE = "NAS_newArch"
        self.ENTRANCE = "search_p_all.py"

        self.DEVICE = 'cuda:0'  # 参数 1
        self.HASHCODE_SIZE = 64 # 参数 2
        self.MAX_ITER = 300     # 参数 3
        self.RESUME_WEIGHT_PATH = None

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
        self.HSIZE = 768            # 参数 4
        self.DROPOUT_R = 0.1
        self.OPS_RESIDUAL = False
        self.OPS_NORM = False
        # Mamba Params
        self.expand_factor: int = 1 # 参数 5
        self.d_model = self.HSIZE   # 参数 6
        self.dt_rank = 4
        self.d_state: int = 16      # 参数 7
        self.d_conv: int = 4
        self.d_inner = self.expand_factor * self.d_model   # 参数 8
        self.dt_min: float = 0.001
        self.dt_max: float = 0.1
        self.dt_init: str = "random"
        self.dt_scale: float = 1.0
        self.dt_init_floor = 1e-4
        self.conv_bias: bool = True
        self.pscan: bool = True     # 参数 9
        self.bias: bool = False
        self.modal_bias: float = 0.8
        # Transformer Params
        self.attention_probs_dropout_prob = 0.1
        self.layer_norm_eps = 1e-12
        self.hidden_dropout_prob = 0.1
        self.intermediate_size = 3072   # 参数 10

        self.NET_LR = 0.001         # 参数 11
        self.ALPHA_LR = 0.1

        #
        self.TWO_OPERATION_RATIO = 3.

        # gene_key
        self.ImgEnc = "ImgEnc"
        self.AudEnc = "AudEnc"
        self.Inter = "Inter"
        self.Fusion = "Fusion"
        self.Final = "Final"

        # logger
        for key, value in vars(self).items():
            log_training_data(f'{key}: {value}', self.LOG_PATH)

        #
        self.writer = SummaryWriter(self.TENSORBOARD_PATH)


class Execution:
    def __init__(self, __C):
        self.__C = __C

    def get_optim(self, net):

        alpha_optim = torch.optim.Adam(net.alpha_prob_parameters(), lr=self.__C.NET_LR)
        net_optim = torch.optim.SGD(net.net_parameters(), self.__C.NET_LR)
        return net_optim,alpha_optim

    def train(self, train_loader, model, criterion, net_optim, alpha_optim, epoch):

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

            # grads = {}
            # for name, param in model.named_alpha_prob_parameters():
            #     grads[name] = torch.autograd.grad(lv, param, retain_graph=True)
            #
            # for name, grad in grads.items():
            #     print(f'Parameter: {name}, Gradient: {grad}')

            net_optim.zero_grad()
            alpha_optim.zero_grad()
            lv.backward()
            net_optim.step()
            alpha_optim.step()
            model_loss += lv.item()

        time_end = time.time() - time_start

        epoch_model_loss = model_loss / len(train_loader)

        split = '-' * 20
        epo = 'Epoch {}/{}'.format(epoch, self.__C.MAX_ITER)
        ml = 'model Loss: {:.6f}'.format(epoch_model_loss)
        t  = 'Train time {:.0f}m {:.0f}s'.format(time_end // 60, time_end % 60)
        info = split + '\n' + epo + '\n' + ml + '\n' + t

        log_training_data(info, self.__C.LOG_PATH)

        self.__C.writer.add_scalar('L_model', epoch_model_loss, global_step=epoch)

    def search(self, train_loader, valid_loader, data_loader):

        model = Net_Search(self.__C)
        model.to(self.__C.DEVICE)

        if self.__C.RESUME_WEIGHT_PATH:
            state_dict = torch.load(self.__C.WEIGHT_PATH, map_location=self.__C.DEVICE)
            model.load_state_dict(state_dict['berthash'], strict=True)
            print('Finish loading ckpt !!!')

        criterion = InfoNCE(negative_mode='paired')

        net_optim, alpha_optim = self.get_optim(model)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     net_optim, self.__C.MAX_ITER, eta_min=self.__C.NET_LR_MIN)

        best_mAP = 0.
        for epoch in range(0, self.__C.MAX_ITER):
            # lr_scheduler.step()
            # lr = lr_scheduler.get_lr()[0]

            # training
            self.train(train_loader, model, criterion, net_optim, alpha_optim, epoch)

            # validation
            mAP = self.validate(valid_loader, data_loader, model)

            info = 'mAP: {:.4f}'.format(mAP)
            log_training_data(info, self.__C.LOG_PATH)
            self.__C.writer.add_scalar('T_mAP', mAP, global_step=epoch)

            # genotype and probs
            genotype1, geneprob1, genotype2, geneprob2 = model.genotype()
            probs = model.genotype_weights()

            if epoch == 0:
                json.dump({}, open(self.__C.ARCH_PATH, 'w+'))
                json.dump({}, open(self.__C.PROB_PATH, 'w+'))
                json.dump({}, open(self.__C.PROBS_PATH, 'w+'))

            formatted_arch_json = json.dumps(combine_genotype_probs(genotype1, geneprob1), indent=4)
            print(formatted_arch_json)

            genotype_json = json.load(open(self.__C.ARCH_PATH, 'r+'))
            genotype_json['epoch' + str(epoch)] = genotype1
            json.dump(genotype_json, open(self.__C.ARCH_PATH, 'w+'), indent=4)

            geneprob_json = json.load(open(self.__C.PROB_PATH, 'r+'))
            geneprob_json['epoch' + str(epoch)] = geneprob1
            json.dump(geneprob_json, open(self.__C.PROB_PATH, 'w+'), indent=4)

            probs_json = json.load(open(self.__C.PROBS_PATH, 'r+'))
            probs_json['epoch' + str(epoch)] = probs
            json.dump(probs_json, open(self.__C.PROBS_PATH, 'w+'), indent=4)


            # genotype as a image
            # plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch + 1))
            # caption = "Epoch {}".format(epoch + 1)
            # plot(genotype.normal, plot_path + "-normal", caption)
            # plot(genotype.reduce, plot_path + "-reduce", caption)

            # save
            if best_mAP < mAP:
                best_mAP = mAP
                best_genotype1 = genotype1
                best_geneprob1 = geneprob1
                best_genotype2 = genotype2
                best_geneprob2 = geneprob2
                best_probs = probs
                state = {
                    'state_dict': model.state_dict(),
                    'net_optim': net_optim.state_dict(),
                }
                best_genotype1['epoch'] = epoch
                json.dump(best_genotype1, open(self.__C.BEST_ARCH_PATH_ONE, 'w+'), indent=4)
                best_geneprob1['epoch'] = epoch
                json.dump(best_geneprob1, open(self.__C.BEST_PROB_PATH_ONE, 'w+'), indent=4)
                best_genotype2['epoch'] = epoch
                json.dump(best_genotype2, open(self.__C.BEST_ARCH_PATH_TWO, 'w+'), indent=4)
                best_geneprob2['epoch'] = epoch
                json.dump(best_geneprob2, open(self.__C.BEST_PROB_PATH_TWO, 'w+'), indent=4)
                best_probs['epoch'] = epoch
                json.dump(best_probs, open(self.__C.BEST_PROBS_PATH, 'w+'), indent=4)

                torch.save(state, self.__C.WEIGHT_PATH)


    def validate(self, test_loader, data_loader, model):

        model.eval()
        with torch.no_grad():
            q_image_code, q_image_targets = self.generate_code(model, test_loader, self.__C.HASHCODE_SIZE,
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

            self.search(train_loader, valid_loader, data_loader)

        elif RUN_MODE in ['val', 'test']:

            # eval_dataset = DataSet(self.__C, RUN_MODE)
            # eval_sampler = SubsetDistributedSampler(eval_dataset, shuffle=False)
            # eval_loader = torch.utils.data.DataLoader(
            #     eval_dataset,
            #     batch_size=self.__C.EVAL_BATCH_SIZE,
            #     sampler=eval_sampler,
            #     num_workers=self.__C.NUM_WORKERS
            # )

            eval_loader = None
            self.eval(eval_loader, valid=RUN_MODE in ['val'])

        else:
            exit(-1)

if __name__ == '__main__':
    __C = CfgSearch()
    exec = Execution(__C)
    exec.run()
