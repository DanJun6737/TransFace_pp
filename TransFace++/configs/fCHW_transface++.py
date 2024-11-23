from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.5, 0.0)   # arcface

config.network = "vit_s_dp005_mask_0"   #ViT-s

config.resume = False
config.output = None
config.embedding_size = 512

config.sample_rate = 1.0
config.fp16 = True
config.weight_decay = 0.1

config.batch_size = 128
config.optimizer = "adamw"

config.lr = 0.001 

config.verbose = 2000000
config.dali = False

config.rec = "/mnt/workspace/danjun/faces_emore"  # MS1MV2
config.num_classes = 85742
config.num_image = 5822653

config.num_epoch = 20 
config.warmup_epoch = config.num_epoch // 10
#config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
config.val_targets = []

config.num_workers = 4
