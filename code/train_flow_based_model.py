import tensorflow as tf
import numpy as np
import os
import sys
import time
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import Model
import pandas as pd

class FlowModel(Model):
    def __init__(self,input_size):
        super(FlowModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(256, activation='relu')
        self.d2 = Dense(256, activation='relu')
        self.d3 = Dense(256, activation='relu')
        self.drop1 = Dropout(0.5)
        self.d4 = Dense(256, activation='relu')
        self.d5 = Dense(256, activation='relu')
        self.drop2 = Dropout(0.5)
        self.d6 = Dense(256, activation='relu')
        self.d7 = Dense(input_size, activation='relu')
        
    def call(self, x,training=None):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.drop1(x,training=training)
        x = self.d4(x)
        x = self.d5(x)
        x = self.drop2(x,training=training)
        x = self.d6(x)
        x = self.d7(x)
        x = tf.clip_by_value(x,0.,1.)
        return x
    
def get_train_ds():
    url_data = '../data/flow_based/Monday-WH-generate-labeled.csv'
    df = pd.read_csv(url_data)
    feats = df.iloc[:,8:]
    ds_port = df.iloc[:,5]
    df = pd.concat([ds_port,feats],axis=1)
    print(df.columns.values)
    all_feats = df.iloc[:,:-1].astype(np.float32).values
    known_data_IDs =(np.any(np.isinf(all_feats),axis=1) + np.any(np.isnan(all_feats),axis=1))==False
    x_train = all_feats[known_data_IDs]
    
    y_train = df.iloc[:,-1].values
    y_train[y_train=='BENIGN']=0.
    y_train = y_train.astype(np.float32)
    y_train = y_train[known_data_IDs]
    
    print(x_train.shape,y_train.shape)
    
    train_min = np.min(x_train,axis=0)
    train_max = np.max(x_train,axis=0)
    
    x_train  = (x_train - train_min)/(train_max - train_min + 1e-6)
    
    return x_train

def make_partial(dset,masked_percent=0.75):
    partial = dset
    mask = np.random.random(partial.shape)
    mask = (mask>masked_percent)*1.
    mask = mask.astype(np.float32)
    partial = partial*mask
    return partial

@tf.function
def repo_loss(rec_x,x):
    MSE_loss = (rec_x - x)**2
    MSE_loss = tf.reduce_mean(MSE_loss)
    return MSE_loss

@tf.function
def train_step(partial_x, x,optimizer):
    with tf.GradientTape() as tape:
        rec_x = model(partial_x, training=True)
        loss = repo_loss(rec_x=rec_x,x=x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)

def train_model(x_train):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    nb_epochs = 5
    batch_size = 256
    total_batch = len(x_train)//batch_size
    if len(x_train) % batch_size!=0:
        total_batch+=1

    start_time = time.time()
    for ep in range(nb_epochs):
        train_loss.reset_states()
        x_train_partial = make_partial(x_train,0.75)
        inds = rng.permutation(x_train_partial.shape[0])
        x_train_partial_perm = x_train_partial[inds]
        x_train_perm = x_train[inds]
        for i in range(total_batch):
            x_batch = x_train_partial_perm[i*batch_size:(i+1)*batch_size]
            y_batch = x_train_perm[i*batch_size:(i+1)*batch_size]
            train_step(x_batch,y_batch,optimizer)
        print ('trained time',time.time() - start_time,ep,'loss:',train_loss.result().numpy())
    print ('total time',time.time() - start_time)


if __name__ == "__main__":
    RANDOM_SEED = 2019
    rng = np.random.RandomState(RANDOM_SEED)
    x_train = get_train_ds()
    input_size = x_train.shape[1]
    
    model = FlowModel(input_size)
    model._set_inputs(tf.TensorSpec([None,input_size]))
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_model(x_train)
    
    model.save('../models/flow_based_model')