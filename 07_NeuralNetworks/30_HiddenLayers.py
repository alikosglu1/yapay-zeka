

# %%
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# %%
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# %%
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


# %%
model.summary()


# %%
# compiling the model
model.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# %%
history = model.fit(x_train, y_train, epochs=30, validation_split=0.2)


# %%
# %matplotlib inline

# print(Y_test[image_index]) # The label is 8
plt.imshow(x_test[5:6, ].reshape(28, 28), cmap='Greys')


# %%
x_test[5:6, ].shape


# %%
predictions = model(x_test[5:6, ]).numpy()
predictions


# %%
tf.nn.softmax(predictions).numpy()


# %%
print(predictions.argmax())


# %%
model.evaluate(x_test,  y_test, verbose=1)


# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(30)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel('Epochs')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.show()


# %%
plt.imshow(x_test[752].reshape(28, 28), cmap='Greys')
pred = model.predict(x_test[752].reshape(1, 28, 28))
print(pred.argmax())

# %%
