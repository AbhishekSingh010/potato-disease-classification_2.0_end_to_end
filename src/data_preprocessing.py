
import tensorflow as tf
# from tensorflow.keras import models,layers
# from tensorflow.keras import models,layers
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf




from backend.logging_config import logger







def get_dataset_partition_tf(ds,train_split=0.8, val_split=0.1,test_split=0.1,shuffle=True,shuffle_size=10000):
    logger.info("get_dataset_partition_tf function get called in src/data_preprocessing.py")
    # """
    
    # """
    ds_size=len(ds)
    train_ds= ds.take(int(ds_size*train_split))
    temp_ds= ds.skip(int(ds_size*train_split))
    val_ds= temp_ds.take(int(ds_size*val_split))
    test_ds=temp_ds.skip(int(ds_size*test_split))
    logger.info(f"function execution compeleted returning trainining datset of length {len(train_ds)} \n and validation datset of length {len(val_ds)} \n and test datset of length {len(test_ds)}")  
    return train_ds,val_ds,test_ds



def preprocess(dataset):
    # dataset = dataimporter()
    logger.info("data importer called compeleted from data_preprocessing")
    
    train_ds,val_ds,test_ds =get_dataset_partition_tf(dataset)
    
    logger.info("get_dataset_partition_tf called compelted from data_preprocessing")
    
    
    
    logger.info("data tuning get started inside preprocess method under file src/preprocess")
    train_ds_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds_ds=val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds_ds=test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds_ds,val_ds_ds,test_ds_ds

