import os
import cv2

from dataset.base_dataset import BaseDataset
import json

class kitti(BaseDataset):
    def __init__(self, data_path, filenames_path='./dataset/filenames/',
                 is_train=True, crop_size=(448, 576), scale_size=None):
        super().__init__(crop_size)

        if crop_size[0] > 480:
            scale_size = (int(crop_size[0]*640/480), crop_size[0])

        self.scale_size = scale_size

        self.is_train = is_train
        self.data_path = os.path.join(data_path, 'kitti_dataset')

        self.image_path_list = []
        self.depth_path_list = []

        txt_path = os.path.join(filenames_path, 'eigen_benchmark')
        if is_train:
            txt_path += '/train_list.txt'
        else:
            txt_path += '/test_list.txt'
 
        self.filenames_list = self.readTXT(txt_path)
        phase = 'train' if is_train else 'test'
        print("Dataset: KITTI")
        print("# of %s images: %d" % (phase, len(self.filenames_list)))

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        # for competition -> img_path = self.filenames_list[idx].split(' ')[0]
        img_path = self.data_path + self.filenames_list[idx].split(' ')[0]
        gt_path = self.data_path + self.filenames_list[idx].split(' ')[1]
        filename = img_path.split('/')[-2] + '_' + img_path.split('/')[-1]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')

        if self.scale_size:
            image = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))
            depth = cv2.resize(depth, (self.scale_size[0], self.scale_size[1]))
                
        height, width = depth.shape
        top_margin = int(height - 352)
        left_margin = int((width - 1216) / 2)
        
        depth = depth[top_margin:top_margin + 352, left_margin:left_margin + 1216]
        image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216]

        if self.is_train:
            image, depth = self.augment_training_data(image, depth)
        else:
            image, depth = self.augment_test_data(image, depth)

        depth = depth / 256.0  # convert in meters
        
        return {'img_path': img_path, 'image': image, 'depth': depth, 'filename': filename, 'class_id': 0}
