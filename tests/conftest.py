import configparser
import json
import os

import pytest

PYTORCH_DATA_LOADER = "tests_mxnet_data_loader"


def pytest_addoption(parser):
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'performance_tests.config.ini'))
    current_config = config["settings"]

    group = parser.getgroup(PYTORCH_DATA_LOADER)

    device = None if current_config.get("device", "None") == "None" else current_config.get("device")
    group.addoption(
        "--device", action="store", default=device,
        help="Specify the cpu or gpu to use . E.g, cpu or gpu:0. By defaults uses GPU if available else uses CPU"
    )

    group.addoption(
        "--numworkers", action="store", default="1", type=int,
        help="Number of workers"
    )

    epochs = current_config.get("epochs", 2)
    group.addoption(
        "--epochs", action="store", default=epochs, type=int,
        help="Number of epochs"
    )

    batch_size = current_config.get("batch_size", 2)
    group.addoption(
        "--batchsize", action="store", default=batch_size, type=int,
        help="Batch size"
    )

    learning_rate = current_config.get("learning_rate", 2)
    group.addoption(
        "--learningrate", action="store", default=learning_rate, type=float,
        help="Learning rate"
    )

    group.addoption(
        "--statsfile", action="store", default="performancetests_mxnet_stats.json",
        help="Stores reuslts about the test"
    )

    loopntimes = current_config.get("loopntimes", 1)
    group.addoption(
        "--loopntimes", action="store", default=loopntimes, type=int,
        help="The number of times to run the timeit for averging"
    )

    group.addoption(
        "--imagesdir", action="store", default="images", type=str,
        help="The directory containing the image files"
    )

    group.addoption(
        "--recordiodirprefix", action="store", type=str, default="images/train",
        help="The directory containing the record files"
    )


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
Pytest custom report..
    :param item:
    :param call:
    """
    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()

    isPrimitive = lambda obj: not hasattr(obj, '__dict__')

    # we only look at actual failing test calls, not setup/teardown
    if rep.when == "call" and rep.passed:
        outfile = item.funcargs["request"].config.getoption("--statsfile")

        mode = "a" if os.path.exists(outfile) else "w"
        stats_obj = {"testname": rep.nodeid}
        with open(outfile, mode) as f:
            # let's also access a fixture for the fun of it
            for a in item.funcargs:
                if isPrimitive(item.funcargs[a]):
                    stats_obj[a] = item.funcargs[a]

            instance_dict = item.funcargs["request"].instance.__dict__
            for a in instance_dict:
                if isPrimitive(instance_dict[a]):
                    key = "request_" + a
                    stats_obj[key] = instance_dict[a]

            f.write(json.dumps(stats_obj) + "\n")
