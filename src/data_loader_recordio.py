import glob
import logging

import mxnet as mx
import os


class DataLoaderRecordIo:
    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __init__(self, idx_path, path_root):
        """

        :param image_type: Either img or rec
        :param imgrecpath: If using image rec files, imagerec path
        :param path_root:
        """
        self.path_root = path_root
        self.idx_path = idx_path

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
        idx_path = os.path.join('.', self.idx_path)

        # data_iter = mx.image.ImageIter(batch_size=batch_size, data_shape=data_shape,
        #                                path_imgrec=os.path.join('.', self.path_root),
        #                                path_imgidx=idx_path)

        data_iter = mx.io.ImageRecordIter(
            path_imgrec=os.path.join('.', self.path_root),
            data_shape=(3, 227, 227),  # output data shape. An 227x227 region will be cropped from the original image.
            batch_size=batch_size,  # number of samples per batch
            #   resize=256 # resize the shorter edge to 256 before cropping
        )

        return data_iter
