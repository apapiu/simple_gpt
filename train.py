import os
import pickle
import requests
import numpy as np
import pandas as pd
import yaml
import sys


import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, Flatten, Dense, Softmax, GlobalAveragePooling1D, Lambda, Add
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import save_model

from IPython.display import clear_output

def get_data():
    folder = ""
    input_file_path = os.path.join(folder, 'input.txt')
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, 'r') as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    def encode(s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(l):
        return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    #TODO - would lower caps everywhere work?

    # create the train and test splits
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    return train_ids, val_ids


def generate_sequences(token_ids, seq_length):
    token_ids = np.array(token_ids, dtype=np.int16)
    n_tokens = len(token_ids)

    X = np.zeros((n_tokens - seq_length, seq_length), dtype=np.int16)
    y = np.zeros((n_tokens - seq_length, seq_length), dtype=np.int16)

    #could also pad with zeros:
    for i in range(0, n_tokens - seq_length):
        X[i] = token_ids[i:i+seq_length]
        y[i] = token_ids[i+1: i+seq_length+1]

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    #add positional encoding:
    pos_enc = np.tile(np.arange(0, seq_length), (X.shape[0], 1))

    return X, pos_enc, y


def sample_top_k(probs, k=1):

    sorted_indices = np.argsort(probs)
    top_k_indices = sorted_indices[-k:]
    top_k_probs = probs[top_k_indices]
    top_k_probs = top_k_probs / np.sum(top_k_probs)
    sampled_index = np.random.choice(top_k_indices, p=top_k_probs)
    return sampled_index


def generate_text(example, nchar=500, k=5, one_char_at_a_time=False):

    for i in range(nchar):
        if one_char_at_a_time:
            clear_output(wait=True)


        if len(example) < seq_length:
            example_np = np.array(example[-seq_length:])
            example_np = pad_sequences([example], maxlen=seq_length)
        else:
            example_np = np.array(example[-seq_length:]).reshape(1, -1)

        positions = np.arange(seq_length).reshape(1,-1)
        probs = model([example_np, positions])[0][-1].numpy()

        #can we plot the probabilities also?
        #preds.sort_values().tail(10).plot(kind="bar")

        next_token = sample_top_k(probs, k)

        example.append(next_token)

        if one_char_at_a_time:
            print(decode(example))

    if not one_char_at_a_time:
        print(decode(example))

class GenerateTextCallback(Callback):
    def __init__(self, example, nchar, 
                 generate_every_n_batches=100, lr_decay=1.5):
        self.example = example
        self.nchar = nchar
        self.batch_counter = 0
        self.generate_every_n_batches = generate_every_n_batches

    def on_epoch_end(self, epoch, logs=None):
        pass
        #hacky:
        #self.example = list(X_val[0])
        #generate_text(self.example, nchar=self.nchar, k=5, one_char_at_a_time=False)

    def on_batch_end(self, batch, logs=None):
        self.batch_counter += 1
        if self.batch_counter % self.generate_every_n_batches == 0:
            self.example = list(X_val[0])
            generate_text(self.example, nchar=self.nchar, k=5, one_char_at_a_time=False)
            lr = self.model.optimizer.lr.numpy()
            print(f"changing learning rate to {lr/lr_decay}")
            model.optimizer.lr.assign(lr/lr_decay)
            


###########
#####MODELS:
###########


def create_triangle_mask(seq_len, n_heads):
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    mask = tf.expand_dims(tf.expand_dims(mask, 0), 0)
    mask = tf.tile(mask, [1, n_heads, 1, 1])  # Repeat the mask for all batches and all heads
    return mask

def attention(qkv):

    q, k, v = qkv
    # should we scale this?
    s = tf.matmul(k, q, transpose_b=True)  # [bs, h*w, h*w]

    # MASKING:
    mask = create_triangle_mask(tf.shape(s)[-1], tf.shape(s)[-3])
    s += mask * -1e9

    beta = tf.nn.softmax(s)  # attention map
    o = tf.matmul(beta, v)  # [bs, h*w, C]
    return o

def reshape_transpose(tensor, d):
        return tf.transpose(tf.reshape(tensor, [-1, tf.shape(tensor)[1], n_heads, d]), [0, 2, 1, 3])

def multi_head_attention_parallel(qkv, n_heads=4):

    q, k, v = qkv

    d_v=d_k=embed_dim//n_heads #internal dimensions 
    q, k, v = map(lambda t, d: reshape_transpose(t, d), [q, k, v], [d_k, d_k, d_v])
    o = attention([q,k,v])

    # Transpose back to concatenate heads: (batch_size, seq_length, n_heads * d_v)
    o = tf.transpose(o, [0, 2, 1, 3])
    concatenated = tf.reshape(o, (tf.shape(o)[0], tf.shape(o)[1], embed_dim))

    return concatenated

def get_simple_model(use_positional_emb=False,
                     attention_layers=0,
                     n_heads=4,
                     dropout=0.1):
    
    input_layer = Input(shape=(seq_length,))
    pos_input = Input(shape=(seq_length,), name="pos_input")

    embedding_layer = Embedding(input_dim=vocab_size,
                                output_dim=embed_dim)(input_layer)

    pos_embedding_layer = Embedding(input_dim=seq_length, output_dim=embed_dim)(pos_input)

    if use_positional_emb:
        x = Add()([embedding_layer, pos_embedding_layer])
    else:
        x = embedding_layer

    for i in range(attention_layers):
        # Attention Block:
        shortcut = x
        q, k, v = Dense(embed_dim)(x), Dense(embed_dim)(x), Dense(embed_dim)(x)
        x = Lambda(multi_head_attention_parallel)([q,k,v], n_heads)
        x = Dense(embed_dim, activation='linear')(x)
        x = Dropout(dropout)(x)
        x = Add()([shortcut, x])
        shortcut = x
        x = Dense(embed_dim, activation='relu')(x)
        x = Dropout(dropout)(x)
        x = Add()([shortcut, x])

    x = Dense(embed_dim//2, activation='relu')(x)
    output_layer = Dense(vocab_size, activation='softmax')(x)

    model = Model(inputs=[input_layer, pos_input], outputs=output_layer)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    model.compile(optimizer="adam", loss=loss_fn)

    return model


if __name__=='__main__':
    ####config
    seq_length = 64
    vocab_size = 65
    embed_dim = 64
    use_positional_emb = True
    attention_layers = 1
    n_heads = 2
    lr = 0.001
    dir_name = f"seq_len{seq_length}{embed_dim}_attenion_{attention_layers}{n_heads}"
    dropout=0.2
    lr_decay = 1.2

    batch_size = 512*2
    epochs = 6

    config = {
        'seq_length': seq_length,
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'use_positional_emb': use_positional_emb,
        'batch_size': batch_size,
        'epochs': epochs,
        'attention_layers': attention_layers,
        'n_head': n_heads
    }
    ### config end

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(f"{dir_name}/config.yaml", 'w') as f:
        yaml.dump(config, f)


    ###script:

    X_train, pos_enc_tr, y_train = generate_sequences(train_ids, seq_length=seq_length)
    X_val, pos_enc_val, y_val = generate_sequences(val_ids, seq_length=seq_length)


    example = list(X_val[0])
    generate_text_callback = GenerateTextCallback(example, nchar=250, 
                                                generate_every_n_batches=1000, lr_decay=lr_decay)

    model = get_simple_model(use_positional_emb, attention_layers, n_heads, dropout)

    print(model.summary())

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    hist = model.fit([X_train, pos_enc_tr],
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=([X_val, pos_enc_val],
                                        y_val),
                    callbacks=[generate_text_callback, tensorboard_callback]
                    )

    pd.DataFrame(hist.history).to_csv(os.path.join(dir_name, "loss.csv"), index=None)