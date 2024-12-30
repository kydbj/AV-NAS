import os


class Path:
    def __init__(self):

        self.DATASET_CONFIG = './json/FCVID.json'           # 修改 1
        self.UP_PATH = '/mnt/add_disk/NAS/FCVID'            # 修改 2

        self.CKPT_PATH = self.CKPT_PATH()
        self.ARCH_PATH = self.CKPT_PATH + 'arch.json'  # 只记录一的
        self.PROB_PATH = self.CKPT_PATH + 'prob.json'   # 只记录一的
        self.PROBS_PATH = self.CKPT_PATH + 'probs.json'
        self.BEST_ARCH_PATH_ONE = self.CKPT_PATH + 'best_arch_one.json' # 记录最好的一
        self.BEST_PROB_PATH_ONE = self.CKPT_PATH + 'best_prob_one.json' # 记录最好的一
        self.BEST_ARCH_PATH_TWO = self.CKPT_PATH + 'best_arch_two.json' # 记录最好的二
        self.BEST_PROB_PATH_TWO = self.CKPT_PATH + 'best_prob_two.json' # 记录最好的二
        self.BEST_PROBS_PATH = self.CKPT_PATH + 'best_probs.json'
        self.WEIGHT_PATH = self.CKPT_PATH + 'weigth.pth'
        self.LOG_PATH = self.CKPT_PATH + 'log.txt'
        self.TENSORBOARD_PATH  = self.CKPT_PATH + '/runs/search'

    def getfileindex(self):
        try:
            files = os.listdir(self.UP_PATH)
            fileindex = sorted([int(x.split('_')[1]) for x in files])[-1]
        except:
            fileindex = 0

        return fileindex + 1

    def CKPT_PATH(self): # /mnt/add_disk/NAS/ActNet/Log_1/a.pth
        ind = self.getfileindex()
        dir_path = self.UP_PATH + '/Log_{}/'.format(ind)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        return dir_path


if __name__ == "__main__":
    path = Path()
    print(path.ARCH_PATH)
    print(path.ARCH_PATH)









