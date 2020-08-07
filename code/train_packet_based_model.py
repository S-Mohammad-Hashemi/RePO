import tensorflow as tf
import numpy as np
import os
import sys
import time
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model



class PacketModel(Model):
    def __init__(self):
        super(PacketModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(2048, activation='relu')
        self.d2 = Dense(num_input*timesteps, activation='relu')

    def call(self, x):
        x = self.flatten(x)
        dense1_out = self.d1(x)
        dense2_out = self.d2(dense1_out)
        d2_out_clip = tf.clip_by_value(dense2_out,0.,1.)
        output = tf.reshape(d2_out_clip,(-1,timesteps,num_input))
        return output

def get_files(day,prefix = '../data/packet_based/'):
    all_files = []
    prefix = prefix+day
    for file in os.listdir(prefix):
        if file.endswith(".npy") and file.startswith('part'):
            all_files.append(os.path.join(prefix, file))
    all_files = sorted(all_files)
    return all_files

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


def get_train_ds():
    train_files = get_files('monday', prefix='../data/packet_based/')
    x_train = []
    for f in train_files:
        print(f)
        x_train.append(np.load(f))
    x_train = np.concatenate(x_train,axis=0)
    x_train = x_train.astype(np.float32)
    x_train_min = np.min(x_train,axis=0)
    x_train_max = np.max(x_train,axis=0)
    x_train_normalized = (x_train - x_train_min)/(x_train_max - x_train_min+0.000001)
    return x_train,x_train_normalized,x_train_min,x_train_max
    

    
def train_model(x_train,x_train_normalized,timesteps,num_input):
    num_iters=10000
    batch_size=512
    last_valid_index = len(x_train_normalized) - timesteps -1 
    start_time = time.time()
    for learning_rate in [0.001, 0.0001, 0.00001]:
        x_train_partial = make_partial(x_train_normalized,0.75)
        print('x_train_partial',x_train_partial.dtype)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        grads = []
        for v in model.trainable_variables:
            grads.append(np.zeros(v.shape))
        optimizer.apply_gradients(zip(grads,model.trainable_variables))

        train_loss.reset_states()
        for n in range(num_iters):
            x_batch = np.zeros((batch_size,timesteps,num_input),dtype=np.float32)
            y_batch = np.zeros((batch_size,timesteps,num_input),dtype=np.float32)
            jj=0
            while jj <len(x_batch):
                r = np.random.randint(0,last_valid_index)
                if np.sum(x_train[r+timesteps-1])==0:
                    continue
                x_batch[jj] = x_train_partial[r:r+timesteps]
                y_batch[jj] = x_train_normalized[r:r+timesteps]
                jj+=1
            train_step(x_batch,y_batch,optimizer)
            if n%1000==0:
                print ('trained time',time.time() - start_time,n,'out of',num_iters,'loss:',train_loss.result().numpy())
    print ('total time',time.time() - start_time)
if __name__ == "__main__":
    
    x_train,x_train_normalized,_,_ = get_train_ds()
    print('Train set shape and type:',x_train_normalized.shape,x_train_normalized.dtype)
    
    timesteps = 20
    num_input = 29
    model = PacketModel()
    model._set_inputs(tf.TensorSpec([None,timesteps,num_input]))
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_model(x_train,x_train_normalized,timesteps,num_input)
    
    model.save('../models/packet_based_model')