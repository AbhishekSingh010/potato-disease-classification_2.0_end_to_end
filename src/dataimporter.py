import tensorflow as tf
from backend.logging_config import logger
import os

def dataimporter():
    logger.info("data importer function called in src/dataimporter.py ")
    # """
    # it will import the data from the data folder 
    # and return the dataset
    
    # """
    IMAGE_SIZE=256
    BATCH_SIZE=32
    dataset= tf.keras.utils.image_dataset_from_directory(
    "data",
    batch_size=32,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    shuffle=True,
    )
    
    logger.info("data import finished returning dataset",dataset)
    return dataset



def class_name(dataset):
    logger.info("classname method getting called in src/dataimporter")
    # """
    # datapreprocessing=>dataset
    # it will return the class name of the dataset
    
    # """
    class_names = dataset.class_names
    logger.info("returrning classname",class_names)
    return class_names      
    
