from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Input

"""
Implementation of Squeeze and Excitation Network in the TensorFlow 2.5.
Paper: https://arxiv.org/pdf/1709.01507.pdf
Blog: https://idiotdeveloper.com/squeeze-and-excitation-networks
"""

def SqueezeAndExcitation(inputs, ratio=8):
    b, _, _, c = inputs.shape
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(c//ratio, activation="relu", use_bias=False)(x)
    x = Dense(c, activation="sigmoid", use_bias=False)(x)
    x = inputs * x
    return x

if __name__ == "__main__":
    inputs = Input(shape=(128, 128, 32))
    y = SqueezeAndExcitation(inputs)
    print(y.shape)
