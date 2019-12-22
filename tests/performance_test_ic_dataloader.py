import configparser
import glob
import logging
import os
import sys
import tempfile
import timeit

import pytest
from mxnet import gluon
from mxnet.gluon.model_zoo import vision as models
from mxnet.optimizer import SGD

from data_loader_images import DataLoaderImages
from data_loader_recordio import DataLoaderRecordIo
from images_utils import get_labels, create_images_list
from mxnet_single_trainer import Trainer


class TestPerformanceTestICDataLoaderLoader():

    @pytest.fixture
    def images_directory(self, request):
        return request.config.getoption("--imagesdir")

    @pytest.fixture
    def record_io_dir_prefix(self, request):
        return request.config.getoption("--recordiodirprefix")

    @pytest.fixture(autouse=True)
    def setUp(self, request):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), 'performance_tests.config.ini'))

        # Set up logging
        logging.basicConfig(level=logging.getLevelName(config["logging"]["level"]),
                            handlers=[logging.StreamHandler(sys.stdout)],
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        self.n_times = 1
        self.num_epochs = request.config.getoption("--epochs")
        self.batch_size = request.config.getoption("--batchsize")
        self.device = request.config.getoption("--device")
        self.num_workers = request.config.getoption("--numworkers")
        learning_rate = request.config.getoption("--learningrate")

        # Optimiser SGD without momentum
        self.optimiser = SGD(momentum=0.9, learning_rate=learning_rate)
        # Softmax cross entropy loss
        self.loss = gluon.loss.SoftmaxCrossEntropyLoss()
        self.data_shape = (3, 224, 224)

    def testPerformanceRawJepgFormat(self, images_directory):
        """Using images in dir {0} pattern {1}"""
        # Arrange
        # Verify images exist

        images_dir = images_directory
        pattern = "{}/**/*.jpg".format(images_dir)
        images_found = len(list(glob.glob(pattern)))
        assert images_found > 10, "At least 10 images required using pattern {} ".format(
            pattern)

        tmpdir = tempfile.mkdtemp()

        images_lst, labels_dict = create_images_list(images_dir)
        data_loader = DataLoaderImages(images_dir, images_lst)
        data_iter = data_loader(self.num_workers, batch_size=self.batch_size, data_shape=self.data_shape)

        # Load network
        net = models.get_model('resnet34_v2', pretrained=False, classes=len(labels_dict))

        # Set up trainer
        sut = Trainer()

        # Act
        result_time = timeit.timeit(
            lambda: sut(data_iter, None, net, self.optimiser, self.loss, tmpdir,
                        batch_size=self.batch_size,
                        epochs=self.num_epochs, ctx=self.device),
            number=self.n_times)

        self.total_train_time = result_time

        print("Total time is {}".format(result_time))

    def testPerformanceRecordIoFormat(self, record_io_dir_prefix):
        """Using images in dir {0} type {2}"""
        # Arrange
        # Verify images exist
        prefix = record_io_dir_prefix

        rec_format = "{}*.rec".format(prefix)
        record_io_files = list(glob.glob(rec_format))
        assert len(record_io_files) > 0, "Atleast one files matching the pattern {} must be found".format(rec_format)

        idx_fprmat = "{}*.idx".format(prefix)
        idx_files = glob.glob(idx_fprmat)
        assert len(idx_files) > 0, "Atleast one files matching the pattern {} must be found".format(idx_fprmat)

        lst_format = "{}*.lst".format(prefix)
        list_files = list(glob.glob(lst_format))
        assert len(list_files) == 1, "Exactly one file matching the pattern {} must be found".format(list_files)

        label_sizes = len(get_labels(list_files[0]))

        tmpdir = tempfile.mkdtemp()

        # TODO: Fix for multiple record io and idx files
        data_loader = DataLoaderRecordIo(idx_files[0], record_io_files[0])
        data_iter = data_loader(self.num_workers, batch_size=self.batch_size, data_shape=self.data_shape)

        # Load network
        net = models.get_model('resnet34_v2', pretrained=False, classes=label_sizes)

        # Set up trainer
        sut = Trainer()

        # Act
        result_time = timeit.timeit(
            lambda: sut(data_iter, None, net, self.optimiser, self.loss, tmpdir,
                        batch_size=self.batch_size,
                        epochs=self.num_epochs),
            number=self.n_times)

        self.total_train_time = result_time
        print("Total time is {}".format(result_time))
