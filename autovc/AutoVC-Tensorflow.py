
"""
    AutoVC-Tensorflow a framework for doing Voice Conversion using Tensorflow 2.
    https://arxiv.org/abs/1905.05879
"""

import os
import pickle

import  matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


dim_neck = 32
dim_embd = 256
dim_pre = 512
freq = 32

# Content Encoder
inputs = tf.keras.Input(shape=(dim_embd+80, 128))
x = inputs
initializer = tf.keras.initializers.GlorotUniform()
x = tf.transpose(x, perm=[0, 2, 1])
for i in range(3):
    x = tf.keras.layers.Conv1D(512, kernel_size=5, strides=1, padding='same', dilation_rate=1, kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
rnn_cells = [tf.keras.layers.LSTMCell(dim_neck) for _ in range(2)]
stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
x = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(stacked_lstm, return_sequences=True))(x)
outputs = x

# informational bottleneck:
output_forward = outputs[:, :, :dim_neck]
output_backward = outputs[:, :, dim_neck:]
codes = []
for i in range(0, outputs.shape[1], freq):
    codes.append(tf.concat((output_forward[:, i+freq-1, :],  output_backward[:, i, :]), axis=1))
encoder_model = tf.keras.Model(inputs=inputs, outputs=codes)

# Decoder
inputs = tf.keras.Input(shape=(128,dim_neck*2+dim_embd,))
initializer = tf.keras.initializers.GlorotUniform()
x = tf.keras.layers.LSTM(dim_pre, return_sequences=True, kernel_initializer=initializer)(inputs)
for i in range(3):
    x = tf.keras.layers.Conv1D(dim_pre, kernel_size=5, strides=1, padding='same', dilation_rate=1, kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
rnn_cells = [tf.keras.layers.LSTMCell(1024) for _ in range(2)]
stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
x = tf.keras.layers.RNN(stacked_lstm, return_sequences=True)(x)
x = tf.keras.layers.Dense(80, kernel_initializer=initializer)(x)
decoder_model = tf.keras.Model(inputs=inputs, outputs=x)

# Postnet:
inputs = tf.keras.Input(shape=(80,128))
x = inputs
x = tf.transpose(x, perm=[0, 2, 1])
initializer = tf.keras.initializers.GlorotUniform()
for i in range(4):
    x = tf.keras.layers.Conv1D(dim_pre, kernel_size=5, strides=1, padding='same', dilation_rate=1, kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('tanh')(x)
x = tf.keras.layers.Conv1D(80, kernel_size=5, strides=1, padding='same', dilation_rate=1, kernel_initializer=initializer)(x)
x = tf.keras.layers.BatchNormalization()(x)
postnet_model = tf.keras.Model(inputs=inputs, outputs=x)



def preprocess_item(item):
    embeddings2 = item[1].reshape((1,256))
    spectrogram = np.load(os.path.join('./data/spmel/',item[2]))[:128, :]
    embeddingsFinal = np.repeat(embeddings2, spectrogram.shape[0], axis=0)
    input_vector = tf.concat([spectrogram, embeddingsFinal], 1)
    return tf.expand_dims(spectrogram,0), tf.expand_dims(tf.transpose(input_vector),0), tf.expand_dims(embeddingsFinal,0) 

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.postnet = postnet_model 
    
    def call(self, input_vector, c_target_embedding):
        codes = self.encoder(input_vector)
        
        if c_target_embedding is None:
            return tf.concat(codes, axis=-1)
        
        reshaped_encoder_output = []
        # up sample
        for code in codes:
            reshaped_encoder_output.append(tf.keras.layers.UpSampling1D(size=32)(tf.expand_dims(code,1)))
        content_encoder_output = tf.concat(reshaped_encoder_output, axis=1)
        decoder_input = tf.concat([content_encoder_output, c_target_embedding], 2)
        # initial reconstruction 
        decoder_output = decoder_model(decoder_input)
        postnet_input = tf.transpose(decoder_output, perm=[0, 2, 1])
        # residual signal
        postnet_output = postnet_model(postnet_input)
        
        # final reconstruction
        mel_outputs_postnet = decoder_output + postnet_output
        mel_outputs_postnet = tf.expand_dims(mel_outputs_postnet, 1)
        decoder_output = tf.expand_dims(decoder_output, 1)
        return decoder_output, mel_outputs_postnet, tf.concat(codes, axis=-1)

def generator_loss(x_real, x_identic, x_identic_psnt, code_real, code_reconst, lambda_cd = 1):

    # Identity mapping loss
#     print('r', x_real.shape)
#     print('d', x_identic.shape)
    
#     print('cr', code_real.shape)
#     print('cd', code_reconst.shape)
    
    g_loss_id = tf.reduce_sum(tf.losses.MSE(x_real, x_identic))   # initial reconstruction loss 
    g_loss_id_psnt = tf.reduce_sum(tf.losses.MSE(x_real, x_identic_psnt))    # final reconstruction loss

    # Code semantic loss.
    g_loss_cd = tf.reduce_sum(tf.abs(code_real - code_reconst)) # content loss
    # Backward and optimize.
    g_loss = g_loss_id + g_loss_id_psnt + lambda_cd * g_loss_cd
    return g_loss


# create dataset object
metaname = "./data/spmel/train.pkl"
meta = pickle.load(open(metaname, "rb"))
datasets = []
for bindx in range(0, len(meta),2):
    spectrogram1, batch1, speaker_embeddings1 = preprocess_item(meta[bindx%(len(meta))])
    spectrogram2, batch2, speaker_embeddings2 = preprocess_item(meta[(bindx+1)%(len(meta))])
    datasets.append((tf.concat([spectrogram1, spectrogram2], 0), tf.concat([batch1, batch2], 0), tf.concat([speaker_embeddings1, speaker_embeddings2],0)))

model = Generator()

optimizer = tf.keras.optimizers.Adam()
num_iters = 1000
loss_values = []
for i in range(num_iters):
    for x_real, input_vector, embeddings in datasets:
        with tf.GradientTape() as tape:
            x_identic, x_identic_psnt, code_real = model(input_vector, embeddings)
            code_reconst = model(input_vector, None)
            loss = generator_loss(tf.expand_dims(x_real,1), x_identic, x_identic_psnt, code_real, code_reconst)
            print(loss.numpy())
            loss_values.append(loss.numpy())
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    if i % 20:
        print(f'Completed {i}th iteration')


plt.figure()
plt.title('Overall Model Loss')
plt.plot(range(num_iters), loss_values)
plt.show()

