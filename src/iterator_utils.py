import os

import mxnet as mx


def get_image_iterator_raw_files(path_root, images_list, batch_size, data_shape):
    """
    Creates mxnet image iterator for raw files. For more details see  https://beta.mxnet.io/api/gluon-related/_autogen/mxnet.image.ImageIter.html

    """

    data_iter = mx.image.ImageIter(batch_size=batch_size, data_shape=data_shape,
                                   imglist=images_list,
                                   path_root=os.path.join('.', path_root))

    return data_iter


def get_image_iterator_rec_files(path_to_rec, batch_size, data_shape):
    """
    Creates mxnet image iterator for record io files files. For more details see  https://beta.mxnet.io/api/gluon-related/_autogen/mxnet.image.ImageIter.html

    """

    data_iter = mx.image.ImageIter(batch_size=batch_size, data_shape=data_shape,
                                   path_imgrec=path_to_rec)

    return data_iter


def get_recordio_iterator(path_to_rec, batch_size, data_shape=(3, 227, 227)):
    """
    Creates mxnet recordio iterator for recordio files. For more details see  https://beta.mxnet.io/api/gluon-related/_autogen/mxnet.image.ImageIter.html

    """

    data_iter = mx.io.ImageRecordIter(
        path_imgrec=path_to_rec,
        data_shape=data_shape,  # output data shape. An 227x227 region will be cropped from the original image.
        batch_size=batch_size,  # number of samples per batch
    )

    return data_iter
