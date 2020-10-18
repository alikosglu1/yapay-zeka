# %%
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt


# %%
mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# %%

X_train, X_test = X_train / 255.0, X_test / 255.0


# %%

# %matplotlib inline
image_index = 2000
print(Y_test[image_index])  # The label is 8
plt.imshow(X_test[image_index], cmap='Greys')


# %%
X_train.shape


# %%
print(Y_test)


# %%
model = tf.keras.models.Sequential([
    # 28*28= 784 şeklide girdi sağlıyor
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='softmax')
])


# %%
# summary of the model
model.summary()


# %%
# compiling the model
model.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# %%
# training the moodel
history = model.fit(X_train, Y_train,
                    batch_size=128, epochs=5,
                    validation_split=0.2)


# %%
# evalute the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)


# %%
# making prediction
predictions = model.predict(X_test)


# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(5)

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
image_index = 4446
plt.imshow(X_test[image_index].reshape(28, 28), cmap='Greys')
pred = model.predict(X_test[image_index].reshape(1, 28, 28))
print(pred.argmax())
print(pred)

# %%
