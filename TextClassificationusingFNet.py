import keras_nlp
import tensorflow as tf
import os
import keras_nlp
from tensorflow import keras

def train_word_piece(ds, vocab_size ):
    word_piece_ds = ds.unbatch().map(lambda x, y: x)
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        word_piece_ds.batch(1000).prefetch(2),
        vocabulary_size=vocab_size,
        reserved_tokens=["[PAD]", "[UNK]"])
    return vocab
 
INTERMEDIATE_DIM = 512
EMBED_DIM = 128

def newFnet(max_features,sequence_length):
    input_ids = keras.Input(shape=(None,), dtype="int64", name="input_ids")

    x = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=max_features,
        sequence_length=sequence_length,
        embedding_dim=EMBED_DIM,
        mask_zero=True,
    )(input_ids)

    x = keras_nlp.layers.FNetEncoder(intermediate_dim= INTERMEDIATE_DIM)(inputs=x)
    x = keras_nlp.layers.FNetEncoder(intermediate_dim= INTERMEDIATE_DIM)(inputs=x)
    x = keras_nlp.layers.FNetEncoder(intermediate_dim= INTERMEDIATE_DIM)(inputs=x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.1)(x)
    outputs = keras.layers.Dense(1)(x)

    return keras.Model(input_ids, outputs, name="fnet_classifier")

NUM_HEADS = 2

def newTransformer(max_features,sequence_length):
    input_ids = keras.Input(shape=(None,), dtype="int64", name="input_ids")

    x = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=max_features,
        sequence_length=sequence_length,
        embedding_dim= EMBED_DIM,
        mask_zero=True )(input_ids)

    x = keras_nlp.layers.TransformerEncoder(intermediate_dim= INTERMEDIATE_DIM, num_heads= NUM_HEADS)(inputs=x)
    x = keras_nlp.layers.TransformerEncoder(intermediate_dim= INTERMEDIATE_DIM, num_heads= NUM_HEADS)(inputs=x)
    x = keras_nlp.layers.TransformerEncoder(intermediate_dim= INTERMEDIATE_DIM, num_heads= NUM_HEADS)(inputs=x)

    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.1)(x)
    outputs = keras.layers.Dense(1)(x)

    return keras.Model(input_ids, outputs, name="transformer_classifier")    
    