import numpy as np
import os
from PIL import Image

class DataLoader:
    def __init__(self, image_dir, target_height = 224, target_width = 224, rgb = True):
        self.data = {}
        # labels
        self.lgood = 0
        self.lbad = 1
        self.ltest = -1

        image_extension = '.jpg'

        # load training data
        dir2label = {'train/good_0': self.lgood, 'train/bad_1': self.lbad, 'test/all_tests': self.ltest}
        for subdir in dir2label:
            folder = os.path.join(image_dir, subdir)
            label = dir2label[subdir]
            images_this_label = []
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                if filename.endswith(image_extension):
                    img = Image.open(filepath)
                    width, height = img.size
                    if width != target_width or height != target_height:
                        print(filepath, '(width, height) resize:(', width, height, ') -> (', target_width, target_height, ')')
                        img = img.resize((target_height, target_width))
                    
                    if rgb:
                        img = img.convert('RGB')
                    else:
                        img = img.convert('L')

                    images_this_label.append(np.array(img))
                else:
                    print(filepath, 'is not a jpg file!')
            self.data[label] = np.asarray(images_this_label)
        
        print('number of good images:', len(self.data[self.lgood]), ' number of bad images:', len(self.data[self.lbad]), ' number of test images:', len(self.data[self.ltest]))

    def get_data(self, train_ratio = 1, shuffle = True):
        if train_ratio > 1 or train_ratio <= 0:
            print('invalid train ratio:', train_ratio)
            return

        data = {}
        num_good = len(self.data[self.lgood])
        num_bad = len(self.data[self.lbad])
        num_test = len(self.data[self.ltest])

        data_test = self.data[self.ltest]
        data_labeled = np.concatenate((self.data[self.lgood], self.data[self.lbad]))
        labels = np.asarray([self.lgood] * num_good + [self.lbad] * num_bad)
        indexes_good = np.arange(0, num_good)
        indexes_bad = np.arange(num_good, len(labels))
        indexes_test = np.arange(num_test)
        
        if shuffle:
            np.random.shuffle(indexes_good)
            np.random.shuffle(indexes_bad)
            np.random.shuffle(indexes_test)
        
        num_train_good = int(num_good * train_ratio)
        num_train_bad = int(num_bad * train_ratio)
        indexes_train = np.concatenate((indexes_good[:num_train_good], indexes_bad[:num_train_bad]))
        indexes_validation = np.concatenate((indexes_good[num_train_good:], indexes_bad[num_train_bad:]))
        if shuffle:
            np.random.shuffle(indexes_train)
            np.random.shuffle(indexes_validation)
        else:
            train_indexes = np.sort(indexes_train)
            validation_indexes = np.sort(indexes_validation)

        data['train_labels'] = labels[indexes_train]
        data['train_data'] = data_labeled[indexes_train]
        data['validation_labels'] = labels[indexes_validation]
        data['validation_data'] = data_labeled[indexes_validation]
        data['test_data'] = data_test[indexes_test]

        return data
        
def is_grey_scale(img):
    # the images are grayscale
    w,h = img.size
    for i in range(w):
        for j in range(h):
            r,g,b = img.getpixel((i,j))
            if r != g != b: return False
    return True


if __name__ == '__main__':
    dataloader = DataLoader('/data0/semi-conductor-image-classification-first')
    train_ratio = 0.7
    data = dataloader.get_data(train_ratio)
    print('train data:', data['train_data'].shape)
    print('validation data:', data['validation_data'].shape)
    print('test data:', data['test_data'].shape)
    print('train labels:', data['train_labels'].shape)
    print('validation labels:', data['validation_labels'].shape)

    # import sys
    # np.set_printoptions(threshold=sys.maxsize)
    # print(data['train_labels'])

    # for i, index in enumerate( np.random.randint(len(data['train_data']), size = 10) ):
    #     # save some sample images
    #     im = Image.fromarray(data['train_data'][index])
    #     print('is grayscale:', is_grey_scale(im))
    #     im.save('test' + str(i) + '.png')
    #     print('test image', i, index, 'label:', data['train_labels'][index])