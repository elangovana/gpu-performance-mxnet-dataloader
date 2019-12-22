# GPU performance comparision of Mxnet dataloader
Sample code snippets for comparing the mxnet performance for loading data

## Prerequisites

1. Python 3.6
2. Virtual environment

## Setup

1. Install dependencies in your virtual environment
    ```bash
    pip install -r src/requirements.txt 
    pip install -r tests/requirements.txt 
    ```



## Dataset

 
1. We will make use of the aircraft.tar ( 28 MB) part of the dataset from [decathlon](http://www.robots.ox.ac.uk/~vgg/decathlon/) . The entire decathalon dataset is  ~388MB in size, but for the tests we will only make use of the aircraft dataset.

    ```bash
    mkdir -p images
    wget http://www.robots.ox.ac.uk/~vgg/share/decathlon-1.0-data.tar.gz 
    tar -xf decathlon-1.0-data.tar.gz 
    tar -xf aircraft.tar -C images
    
    ```
    
    You should the aircraft directory in the images folder. Within the aircraft directory you will see the train, test and val directories.



### Scenario: Process raw files using mxnet image iterator


1. Run tests using raw images and the stats will be saved in "Stats.json" 
    
    ```bash
    
    export PYTHONPATH=.:tests:./src
 
    pytest   --durations=0  --show-capture=all  --log-cli-level=WARN --show-capture=all -k 'testPerformanceRawJepgFormat'  tests/performance_test_ic_dataloader.py --loopntimes 5 --epochs 50  --batchsize 32 --imagesdir images/aircraft/train --statsfile "Stats.json" 

    ```

### Scenario: Process record io files using mxnet image iterator

1. Create RecordIO formatted files

   ```bash
   
    # Create directory to hold
    mkdir -p ./tests/recordio/train 
    
    # download im2rec.py from mxnet to convert image files to record io
    wget https://raw.githubusercontent.com/apache/incubator-mxnet/1a7199691f5cbc6012bb53eecbf884bed5ae6590/tools/im2rec.py
    
    # Create list file
    python im2rec.py --list --recursive  ./tests/recordio/train  ./tests 
    
    # Create the record io files
    python im2rec.py --recursive   --pass-through --pack-label --num-thread 8 ./tests/recordio/train.lst  ./tests/images 
    
    ```
    
1. Run tests using record io ImageRecordIter formatter and the stats will be saved in "Stats.json" 
    
    ```bash
    
    export PYTHONPATH=.:tests:./src
 
    pytest   --durations=0  --show-capture=all  --log-cli-level=WARN --show-capture=all -k 'testPerformanceRecordIoIteratorRecordIoFormat'  tests/tests_mxnet_data_loader/performance_test_ic_dataloader.py --loopntimes 5 --epochs 50  --batchsize 32 --recordiodirprefix tests/tests_mxnet_data_loader/images/train --statsfile "Stats.json" 

    ```