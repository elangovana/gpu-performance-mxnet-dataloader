import logging
import time

import mxnet as mx
from mxnet import autograd as ag
from mxnet import gluon


class Trainer:
    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __call__(self, train_data, test_data, net, optimiser, loss, model_dir, batch_size=128, epochs=10, ctx=None):
        if ctx is None:
            ctx = [mx.gpu(0)] if mx.context.num_gpus() > 0 else [mx.cpu()]

        batch_size *= max(1, len(ctx))

        # Collect all parameters from net and its children, then initialize them.
        net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)

        # Trainer is for updating parameters with gradient.
        trainer = gluon.Trainer(net.collect_params(), optimiser)
        metric = mx.metric.Accuracy()

        best_accuracy = 0.0
        for epoch in range(epochs):
            # reset data iterator and metric at begining of epoch.
            train_data.reset()
            tic = time.time()
            metric.reset()
            btic = time.time()

            for i, batch in enumerate(train_data):
                data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
                outputs = []
                Ls = []
                with ag.record():
                    for x, y in zip(data, label):
                        z = net(x)
                        L = loss(z, y)
                        # store the loss and do backward after we have done forward
                        # on all GPUs for better speed on multiple GPUs.
                        Ls.append(L)
                        outputs.append(z)
                    for L in Ls:
                        L.backward()
                trainer.step(batch.data[0].shape[0])
                metric.update(label, outputs)

                name, acc = metric.get()
                logging.info('Epoch [%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f' %
                             (epoch, i, batch_size / (time.time() - btic), name, acc))
                btic = time.time()

            name, acc = metric.get()
            logging.info('[Epoch %d] training: %s=%f' % (epoch, name, acc))
            logging.info('[Epoch %d] time cost: %f' % (epoch, time.time() - tic))

            # name, val_acc = self.test(ctx, net, test_data)
            # logging.info('[Epoch %d] validation: %s=%f' % (epoch, name, val_acc))
            #
            # # only save params on primary host
            # if current_host == hosts[0]:
            #     if val_acc > best_accuracy:
            #         net.save_params('{}/model-{:0>4}.params'.format(model_dir, epoch))
            #         best_accuracy = val_acc

        return net

    def test(self, ctx, net, test_data):
        test_data.reset()
        metric = mx.metric.Accuracy()

        for i, batch in enumerate(test_data):
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            for x in data:
                outputs.append(net(x))
            metric.update(label, outputs)
        return metric.get()
