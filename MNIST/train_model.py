import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl

model_json_path = 'models/model.json'
weights_path = 'models/model.h5'

train_X = np.load('./preprocessed/train_X.npy')
train_y = np.load('./preprocessed/train_y.npy')

print(train_X.shape)
print(train_y.shape)

m, n_xh, n_xw, n_xc = train_X.shape

X = tfl.Input(shape=(n_xh, n_xw, n_xc))

# Block 1
out1 = tfl.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', kernel_initializer='he_normal')(X)
out1 = tfl.MaxPooling2D((2, 2))(out1)

# Block 2
out2 = tfl.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(out1)
out2 = tfl.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(out2)

out1 = tfl.Conv2D(64, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='he_normal')(out1)
out2 = tfl.Add()([out1, out2])

# Block 3
out3 = tfl.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(out2)
out3 = tfl.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(out3)

out2 = tfl.Conv2D(128, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='he_normal')(out2)
out3 = tfl.Add()([out2, out3])
out3 = tfl.MaxPooling2D((2, 2))(out3)

out3 = tfl.Flatten()(out3)
out = tfl.Dense(300, kernel_initializer='he_normal')(out3)
out = tfl.BatchNormalization()(out)
out = tfl.Dropout(0.3)(out)
out = tfl.Activation('relu')(out)

out = tfl.Dense(200, kernel_initializer='he_normal')(out)
out = tfl.BatchNormalization()(out)
out = tfl.Dropout(0.3)(out)
out = tfl.Activation('relu')(out)

out = tfl.Dense(100, kernel_initializer='he_normal')(out)
out = tfl.BatchNormalization()(out)
out = tfl.Dropout(0.3)(out)
out = tfl.Activation('relu')(out)

out3 = tfl.Dense(100, kernel_initializer='he_normal')(out3)
out3 = tfl.BatchNormalization()(out3)
out3 = tfl.Dropout(0.3)(out3)
out3 = tfl.Activation('relu')(out3)

out = tfl.Add()([out, out3])

out = tfl.Dense(10, activation='softmax')(out)

model = tf.keras.Model(inputs=X, outputs=out)

model.summary()

print('Fitting...')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=tf.keras.metrics.sparse_categorical_accuracy)

data_fit = model.fit(train_X, train_y, epochs=10, batch_size=200)

print(data_fit.history)

print('Saving model...')
with open(model_json_path, 'wt') as file:
    file.write(model.to_json())

print('Saving weights...')
model.save_weights(weights_path)





