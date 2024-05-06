
import tensorflow as tf
import numpy as np
import os

from src.dataimporter import dataimporter,class_name
from backend.logging_config import logger

import streamlit as st


def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr  # Keep the initial learning rate for the first 10 epochs
    else:
        return lr * tf.math.exp(-0.1) 













def model_training(train_ds,val_ds,test_ds,epochs=10,batch_size=32,learning_rate=0.001):
    logger.info('model training started')
    st.write(f'The number of epochs set to: {epochs}')
    st.write(f'The number of batch_size set to:{batch_size}')
    st.write(f'The number of learning rate set to: {learning_rate}')
    
    
    IMAGE_SIZE=256
    BATCH_SIZE=batch_size
    CHANNELS=3
    EPOCHS=epochs
    
    # train_ds,val_ds,test_ds=preprocess() 
    
    resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    tf.keras.layers.experimental.preprocessing.Rescaling(1.0/255)
])
   # data augmentation

    data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])
   
    # model building - cnn
    input_shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    
    # n_classes=3
    dataset=dataimporter()
    n_classes=len(class_name(dataset))
  
    model=tf.keras.Sequential([
    resize_and_rescale,
    data_augmentation,
    tf.keras.layers.Conv2D(32,(3,3),activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(n_classes,activation='softmax'),
])

    model.build(input_shape=input_shape)
    logger.info(f"model summary is{model.summary}")
    
    initial_lr = learning_rate  # Set your initial learning rate here
    opt = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    model.compile(
    optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

    # st.write(f"the model architecture is {model.summary}")
    
    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    
    
    history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds,
    callbacks=[lr_scheduler_callback]  # Pass the callback to the fit method
)
    
  
    model_version=max([int (i)for i in os.listdir("./models")+[0]])+1
    model.save(f"./models/{model_version}")
    logger.info("Model saved")
    st.subheader(f"saved_model name is =>{model_version} ")

    return history
    
    
    
