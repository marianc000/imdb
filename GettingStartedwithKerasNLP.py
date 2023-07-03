import keras_nlp
from tensorflow.keras import layers
from tensorflow import keras
# only with gpu
keras.mixed_precision.set_global_policy("mixed_float16")

# classifier = keras_nlp.models.BertClassifier.from_preset(bertModel,num_classes=2, preprocessor=None)
# classifier.summary()

def newModel(bertModel):
    backbone = keras_nlp.models.BertBackbone.from_preset(bertModel)
     
    inputs = backbone.input
    sequence = backbone(inputs)["sequence_output"]
     
    # Use [CLS] token output to classify
    x =  layers.Dropout(0.1)(sequence[:, backbone.cls_token_index, :])
    outputs =  layers.Dense(1)(x)
    
    return keras.Model(inputs, outputs)