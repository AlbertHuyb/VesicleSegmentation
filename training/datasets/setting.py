from easydict import EasyDict as edict

# init
__C_Vesicles = edict()

cfg_data = __C_Vesicles

__C_Vesicles.STD_SIZE = (768,1024)
__C_Vesicles.TRAIN_SIZE = (576,768)
__C_Vesicles.DATA_PATH = '/mnt/pfister_lab2/yubin/vesiclesNew/Vesicles2/'               

__C_Vesicles.MEAN_STD = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])

__C_Vesicles.LABEL_FACTOR = 1
__C_Vesicles.LOG_PARA = 100.

__C_Vesicles.RESUME_MODEL = ''#model path
__C_Vesicles.TRAIN_BATCH_SIZE = 4 #imgs

__C_Vesicles.VAL_BATCH_SIZE = 4 # 


