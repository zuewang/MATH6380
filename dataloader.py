import numpy as np
import os
from PIL import Image

class DataLoader:
    def __init__(self, image_dir):
        '''
        The labels provided is .docx which cannot be read by python directly
        so just set a label vector here
        0: Not Raphael
        1: Raphael
        2: Disputed
        '''
        self.labels = np.array([2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 2, 2, 1, 1])
        self.size = self.labels.shape[0]
        self.root_dir = image_dir

        self.raw_data = {}
        self.transformed_data = []

        image_extensions = ['jpg', 'TIF', 'tif', 'jpeg', 'tiff', 'jpg']
        
        candidate_indexes = list(range(0, self.size))

        for name in os.listdir(image_dir):
            ext = name.split('.')[-1]
            if ext in image_extensions:
                try:
                    image_index = int(name.split('.')[0]) - 1
                    candidate_indexes.remove(image_index)
                    img = Image.open(os.path.join(image_dir, name))
                    self.raw_data[image_index] = {'filename': name, 'data': img, 'label': self.labels[image_index]}
                except:
                    print('not related to this project:', name)
        
        if len(candidate_indexes) > 0:
            print('[ERROR] These image indexes do not exist:', candidate_indexes)
            return

    def transform(self, target_height = 224, target_width = 224, grayscale = True):
        # resize image to target size (not crop or padding)
        for index in sorted(self.raw_data.keys()):
            img = self.raw_data[index]['data']
            resized = img.resize((target_height, target_width))
            if grayscale:
                resized = resized.convert('L')
            self.transformed_data.append(np.array(resized))
        self.transformed_data = np.asarray(self.transformed_data)
        # TODO: rotate height, width, dimension? tiff has 4 dimensions, will cause error here

    def get_data(self, train_ratio = 1, shuffle = True):
        if train_ratio > 1 or train_ratio <= 0:
            print('invalid train ratio:', train_ratio)
            return

        data = {}
        indexes0 = np.where(self.labels == 0)[0] # np.where return a tuple containing the array
        indexes1 = np.where(self.labels == 1)[0]
        test_indexes = np.where(self.labels == 2)[0]
        num_train0 = int(len(indexes0) * train_ratio)
        num_train1 = int(len(indexes1) * train_ratio)
        if shuffle:
            np.random.shuffle(indexes0)
            np.random.shuffle(indexes1)
            np.random.shuffle(test_indexes)

        train_indexes = np.concatenate((indexes0[:num_train0], indexes1[:num_train1]))
        validation_indexes = np.concatenate((indexes0[num_train0:], indexes1[num_train1:]))
        if shuffle:
            np.random.shuffle(train_indexes)
            np.random.shuffle(validation_indexes)
        else:
            train_indexes = np.sort(train_indexes)
            validation_indexes = np.sort(validation_indexes)
        # np.dtype?
        
        data['train_labels'] = self.labels[train_indexes]
        data['train_data'] = self.transformed_data[train_indexes]
        data['validation_labels'] = self.labels[validation_indexes]
        data['validation_data'] = self.transformed_data[validation_indexes]
        data['test_data'] = self.transformed_data[test_indexes]

        return data
        



if __name__ == '__main__':
    dataloader = DataLoader('/data0/Downloads/Raphael Project final copy')
    dataloader.transform()
    data = dataloader.get_data()
    print(data['train_data'].shape)
    print(data['validation_data'].shape)
    print(data['test_data'].shape)
    print(data['train_labels'].shape)
    print(data['validation_labels'].shape)