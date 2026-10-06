import os
import pickle
import shutil
import glob
import datetime


class train_config:
    def __init__(self, name,batch_size  ,resize ):
        self.adam_betas = (0.9, 0.999)
        self.weight_decay = 1e-6
        self.scheduler_step = 5
        self.scheduler_gamma = 0.5
        self.learning_rate = 2.5e-3
        # self.datalabel = datalabel
        self.resize = resize
        self.normalize = dict(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5])
        self.max_imgs = None  # 默认不限制
        self.workers = 4
        self.batch_size = batch_size


        # === 根据 datalabel 设置默认值 ===
        if 'ff-'or 'else' in name:
            if 'ff-5' in name:
                self.num_classes = 5
            self.datalabel = name
            self.imgs_per_video = 50
            self.frame_interval = 10
            self.max_frames = 500
            self.augment = 'augment0'
        elif 'dfdc' in name:
            self.datalabel = 'dfdc'
            self.max_frames = 300
            self.imgs_per_video = 30
            self.frame_interval = 10
        elif 'celebv2' in name:  # CelebDF
            self.datalabel = 'celebv2'
            self.max_frames = 300
            self.imgs_per_video = 30
            self.frame_interval = 10
        elif 'dfw' in name:  #
            self.datalabel = 'dfw'
            self.max_frames = 400
            self.imgs_per_video = 30
            self.frame_interval = 10

        elif 'celebv1' in name:  # DFDC
            self.datalabel = 'celebv1'
            self.max_frames = 300
            self.imgs_per_video = 30
            self.frame_interval = 10
        elif 'faceshifter' in name:  # DFDC
            self.datalabel = 'faceshifter'
            self.max_frames = 300
            self.imgs_per_video = 30
            self.frame_interval = 10

        self.dataset = dict(
            datalabel=self.datalabel,
            resize=self.resize,
            normalize=self.normalize,
            augment='augment0',
            max_imgs=self.max_frames
        )