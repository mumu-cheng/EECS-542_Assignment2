import os
import numpy as np
import random
import pdb

from PIL import Image

class voc_reader:
    def __init__(self, data_path, mode):
        self.dir = data_path
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.batch_offset = 0
        self.mode = mode
        self.__load_data()

    def __load_image(self, idx):
        im = Image.open('{}/JPEGImages/{}.jpg'.format(self.dir, idx))
        im = np.array(im.getdata()).astype(np.float32).reshape((im.size[0],im.size[1],3))
	im = im[np.newaxis, ...]
	return im

    def __load_label(self, idx):
        label = Image.open('{}/SegmentationClass/{}.png'.format(self.dir, idx))
        label = np.array(label, dtype=np.uint8)
        label = label[np.newaxis, ...]
        label = label[..., np.newaxis]
	return label

    def __load_data(self):
        # read in indices of training instances
        indices_file = '{}/ImageSets/Segmentation/{}.txt'.format(self.dir, self.mode)
        self.data_indices = open(indices_file, 'r').read().splitlines()

        self.images = []
        self.labels = []
        cnt = 1
        for idx in self.data_indices:
            if cnt % 100 == 0:
                print 'progress: {}/{}'.format(cnt, len(self.data_indices))
            	break
	    self.images.append(self.__load_image(idx))
            self.labels.append(self.__load_label(idx))
            cnt = cnt + 1

        self.MAX_BATCHSIZE = len(self.images)
        return

    def get_next_pair(self):
        cur_offset = self.batch_offset
        self.batch_offset += 1 # batch_size is fixed to be one
        
        if self.batch_offset >= self.MAX_BATCHSIZE:

            # shuffle the dataset
            random.shuffle(self.images)
            random.shuffle(self.labels)

            # start next epoch
            self.batch_offset = 0

        return self.images[cur_offset], self.labels[cur_offset]
