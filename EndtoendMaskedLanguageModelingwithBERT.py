from dataclasses import dataclass
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from tensorflow import keras


@dataclass
class Config:
    LR = 0.001
    EMBED_DIM = 128
    NUM_HEAD = 8  # used in bert model
    FF_DIM = 128  # used in bert model
    NUM_LAYERS = 1

config = Config()

def bert_module(query, key, value, i):
    # Multi headed self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=config.NUM_HEAD,
        key_dim=config.EMBED_DIM // config.NUM_HEAD,
        name="encoder_{}/multiheadattention".format(i))(query, key, value)
    attention_output = layers.Dropout(0.1, name="encoder_{}/att_dropout".format(i))(
        attention_output
    )
    attention_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/att_layernormalization".format(i)
    )(query + attention_output)

    # Feed-forward layer
    ffn = keras.Sequential(
        [
            layers.Dense(config.FF_DIM, activation="relu"),
            layers.Dense(config.EMBED_DIM),
        ],
        name="encoder_{}/ffn".format(i),
    )
    ffn_output = ffn(attention_output)
    ffn_output = layers.Dropout(0.1, name="encoder_{}/ffn_dropout".format(i))(
        ffn_output
    )
    sequence_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/ffn_layernormalization".format(i)
    )(attention_output + ffn_output)
    return sequence_output
 
def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc

def newBert(sequence_length,vocablaryLength):
    inputs = layers.Input((sequence_length,), dtype=tf.int64)
    word_embeddings = layers.Embedding(vocablaryLength, config.EMBED_DIM, name="word_embedding")(inputs)
    position_embeddings = layers.Embedding( input_dim=sequence_length,output_dim=config.EMBED_DIM,
        weights=[get_pos_encoding_matrix(sequence_length, config.EMBED_DIM)], name="position_embedding",
    )(tf.range(start=0, limit=sequence_length, delta=1))
    embeddings = word_embeddings + position_embeddings

    encoder_output = embeddings
    
    for i in range(config.NUM_LAYERS):
        encoder_output = bert_module(encoder_output, encoder_output, encoder_output, i)

    pooled_output = layers.GlobalMaxPooling1D()(encoder_output)
    hidden_layer = layers.Dense(64, activation="relu")(pooled_output)
    outputs = layers.Dense(1 )(hidden_layer)
    return keras.Model(inputs, outputs, name="classification")