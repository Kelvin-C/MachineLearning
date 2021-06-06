import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl

model_json_path = 'models/model.json'
weights_path = 'models/model.h5'

train_X = np.load('./preprocessed/train_X.npy')
train_Y = np.load('./preprocessed/train_y.npy')
dev_X = np.load('./preprocessed/dev_X.npy')
dev_Y = np.load('./preprocessed/dev_Y.npy')

print(f'Train_X shape: {train_X.shape}')
print(f'Train_Y shape: {train_Y.shape}')
print(f'dev_X shape: {dev_X.shape}')
print(f'dev_Y shape: {dev_Y.shape}')

m, n_x = train_X.shape

X = tfl.Input(shape=(n_x, ))


def dense_block(input_layer: tfl.Layer) -> tfl.Layer:
    block = tfl.Dense(200, kernel_initializer='he_normal')(input_layer)
    block = tfl.BatchNormalization()(block)
    block = tfl.Dropout(0.5)(block)
    return tfl.Activation('relu')(block)


out = dense_block(X)
out = dense_block(out)
shortcut = out
out = dense_block(out)
out = dense_block(out)
out = dense_block(out)
out = tfl.Add()([shortcut, out])


out = tfl.Dense(1, activation=tf.keras.activations.sigmoid)(out)

model = tf.keras.Model(inputs=X, outputs=out)

model.summary()

print('Fitting...')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=tf.keras.metrics.binary_accuracy)

data_fit = model.fit(train_X, train_Y, validation_data=(dev_X, dev_Y), epochs=100, batch_size=256)

print('Saving model...')
with open(model_json_path, 'wt') as file:
    file.write(model.to_json())

print('Saving weights...')
model.save_weights(weights_path)





