
import logging

import mxnet as mx
import os


class DataLoaderImages:
    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __init__(self, path_root, images_list):
        """
        :param path_root:
        """
        self.images_list = images_list
        self.path_root = path_root

    def __call__(self, num_workers, batch_size, data_shape,
                 resize=-1,
                 num_parts=1, part_index=0):
        """
Load data https://mxnet.incubator.apache.org/versions/master/tutorials/basic/data.html

        :param path: Image path
        :param augment:
        :param num_workers: num of workers to use
        :param batch_size:
        :param data_shape:
        :param resize:
        :param num_parts:
        :param part_index:
        :return:

        """

        data_iter = mx.image.ImageIter(batch_size=batch_size, data_shape=data_shape,
                                       imglist=self.images_list,
                                       path_root=os.path.join('.', self.path_root))

        return data_iter
