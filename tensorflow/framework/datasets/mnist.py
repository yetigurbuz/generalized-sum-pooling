import tensorflow as tf

import numpy as np

import os

from .dataset_base import Dataset

import PIL

class MNIST(Dataset):

    def _reOrganizeDataset(self, base_dir):

        def saveImages(target_path, images, labels, batch_id):

            file_names = []

            num_data = len(labels)

            prog_bar = tf.keras.utils.Progbar(target=num_data)
            for k in range(num_data):

                '''if k % 1000 == 0:
                    print('INFO:Dataset:{}:saving ({}) / ({})'.format('images', k + 1, num_data))'''


                img_name = os.path.join(target_path,
                                        'c%02d'%(labels[k] + 1),
                                        '%02d_%05d_%02d.png'%(batch_id + 1, k + 1, labels[k] + 1))

                with tf.device('CPU:0'):
                    if not tf.io.gfile.exists(img_name):
                        img = np.expand_dims(images[k], -1)
                        img_decoded = tf.image.encode_png(img)
                        with tf.io.gfile.GFile(img_name, 'bw') as f:
                            f.write(img_decoded.numpy())

                file_names.append(img_name)

                prog_bar.add(1)

            return file_names

        base_dir = os.path.join(os.path.abspath(os.getcwd()), base_dir)

        # original dataset organization
        zipped_dataset_file = os.path.join(base_dir, 'mnist.npz')
        if not os.path.exists(zipped_dataset_file):
            print(self.information('Downloading and extracting the datasets...'))

        (train_data, train_labels), (test_data, test_labels) = \
            tf.keras.datasets.mnist.load_data(os.path.abspath(zipped_dataset_file))

        num_classes = {'train': 10, 'val': 0, 'eval': 10}

        # create related folders for the reorganizaton (create datasets with raw images)
        path_to_train_images = os.path.join(base_dir, 'images', 'training')
        path_to_test_images = os.path.join(base_dir, 'images', 'evaluation')

        if not tf.io.gfile.exists(path_to_train_images):
            tf.io.gfile.makedirs(path_to_train_images)
            [tf.io.gfile.makedirs(os.path.join(path_to_train_images, 'c%02d'%(i + 1))) for i in range(num_classes['train'])]
            print(self.information('Created class folders for %s'%path_to_train_images))

        if not tf.io.gfile.exists(path_to_test_images):
            tf.io.gfile.makedirs(path_to_test_images)
            [tf.io.gfile.makedirs(os.path.join(path_to_test_images, 'c%02d'%(i + 1))) for i in range(num_classes['eval'])]
            print(self.information('Created class folders for %s' % path_to_test_images))


        # compute datasets mean to be used in preprocessing later
        dataset_mean = np.mean(train_data, axis=0).astype(np.float32) / 255

        # now write raw images
        example_names = {}

        print(self.information('Writing training images'))
        example_names['train'] = saveImages(path_to_train_images, train_data, train_labels, 0)

        example_names['val'] = None

        print(self.information('Writing evaluation images'))
        example_names['eval'] = saveImages(path_to_test_images, test_data, test_labels, 2)

        shard_size = 10000

        return example_names, num_classes, shard_size, dataset_mean.tolist(), 1.


    def _exampleFeatures(self, example_name):
        # returns a list of arguments for tf examples extracted from the data of example_name

        image_path = example_name
        label = int(image_path.split('_')[-1].split('.')[0])

        image = PIL.Image.open(image_path, 'r')
        height = image.height
        width = image.width

        arg_list = [{'image_path': image_path, 'label': label - 1, 'height': height, 'width': width}]

        return arg_list

class MNISTML(Dataset):

    def _reOrganizeDataset(self, base_dir):

        def saveImages(target_path, images, labels, batch_id, label_map):

            file_names = []

            num_data = len(labels)

            prog_bar = tf.keras.utils.Progbar(target=num_data)
            for k in range(num_data):

                '''if k % 1000 == 0:
                    print('INFO:Dataset:{}:saving ({}) / ({})'.format('images', k + 1, num_data))'''

                label = label_map[labels[k]]
                img_name = os.path.join(target_path,
                                        'c%02d'%label,
                                        '%02d_%05d_%02d.png'%(batch_id + 1, k + 1, label))

                with tf.device('CPU:0'):
                    if not tf.io.gfile.exists(img_name):
                        img = np.expand_dims(images[k], -1)
                        img_decoded = tf.image.encode_png(img)
                        with tf.io.gfile.GFile(img_name, 'bw') as f:
                            f.write(img_decoded.numpy())

                file_names.append(img_name)

                prog_bar.add(1)

            return file_names

        # original dataset organization
        zipped_dataset_file = os.path.join(base_dir, 'mnist.npz')
        if not os.path.exists(zipped_dataset_file):
            print(self.information('Downloading and extracting the datasets...'))

        # raw train test split
        (train_data, train_labels), (test_data, test_labels) = \
            tf.keras.datasets.mnist.load_data(zipped_dataset_file)
        # splits for zero shot setting
        train_label_subset = [1, 3, 5, 7, 9]
        test_label_subset = [0, 2, 4, 6, 8]
        train_label_map = {k: v + 1 for v, k in enumerate(train_label_subset)}
        test_label_map = {k: v + 1 for v, k in enumerate(test_label_subset)}

        all_data = np.concatenate([train_data, test_data], axis=0)
        all_labels = np.concatenate([train_labels, test_labels], axis=0)

        split_ids = [[], []]

        [split_ids[int(lbl in test_label_subset)].append(id) for id, lbl in enumerate(all_labels)]

        train_ids, test_ids = split_ids
        train_data = all_data[train_ids]
        train_labels = all_labels[train_ids]
        test_data = all_data[test_ids]
        test_labels = all_labels[test_ids]

        num_classes = {'train': 5, 'val': 0, 'eval': 5}

        # create related folders for the reorganizaton (create datasets with raw images)
        path_to_train_images = os.path.join(base_dir, 'images', 'training')
        path_to_test_images = os.path.join(base_dir, 'images', 'evaluation')

        if not tf.io.gfile.exists(path_to_train_images):
            tf.io.gfile.makedirs(path_to_train_images)
            [tf.io.gfile.makedirs(os.path.join(path_to_train_images, 'c%02d'%(i + 1))) for i in range(num_classes['train'])]
            print(self.information('Created class folders for %s'%path_to_train_images))

        if not tf.io.gfile.exists(path_to_test_images):
            tf.io.gfile.makedirs(path_to_test_images)
            [tf.io.gfile.makedirs(os.path.join(path_to_test_images, 'c%02d'%(i + 1))) for i in range(num_classes['eval'])]
            print(self.information('Created class folders for %s' % path_to_test_images))


        # compute datasets mean to be used in preprocessing later
        dataset_mean = np.mean(train_data, axis=0).astype(np.float32) / 255

        # now write raw images
        example_names = {}

        print(self.information('Writing training images'))
        example_names['train'] = saveImages(path_to_train_images, train_data, train_labels, 0, train_label_map)

        example_names['val'] = None

        print(self.information('Writing evaluation images'))
        example_names['eval'] = saveImages(path_to_test_images, test_data, test_labels, 2, test_label_map)

        shard_size = 10000

        return example_names, num_classes, shard_size, dataset_mean.tolist()


    def _exampleFeatures(self, example_name):
        # returns a list of arguments for tf examples extracted from the data of example_name

        image_path = example_name
        label = int(image_path.split('_')[-1].split('.')[0])

        image = PIL.Image.open(image_path, 'r')
        height = image.height
        width = image.width

        arg_list = [{'image_path': image_path, 'label': label - 1, 'height': height, 'width': width}]

        return arg_list

