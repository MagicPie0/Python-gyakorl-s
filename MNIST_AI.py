from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, Input
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt 
import numpy as np

print("Betöltés indul")
#Betöltjük az adatokat
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Betöltött")

print("Train shape: ", x_train.shape, y_train.shape)
print("Test shape: ", x_test.shape, y_test.shape)

#Normalizálás 0-1 közé
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

#Extra dimenzó a CNN miatt kell channels=1, mert szürkeárnyalatos
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("Train: ", x_train.shape, y_train.shape)
print("Test: ", x_test.shape, y_test.shape)

#Címkék one-hot encodingja (0-9 számjegyek)
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

model = Sequential(
    [
        Flatten(input_shape=(28, 28, 1)), # A 28*28-as kép lapítása 784-elemre
        Dense(128, activation='relu'), #Rejtett réteget képezünk a neurális hálón 128 neuronnal
        Dense(10, activation='softmax') # Kiadunk 10 osztályt (0-9)
    ]
)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    x_train,
    y_train_cat,
    epochs=5,
    batch_size=32,
    validation_split=0.2 #20% validálás
)

#Teszt halmaz pontossága
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"Teszt pontossága: {test_acc:.4f}")

model_cnn = Sequential(
    [
        Input(shape=(28,28,1)),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), #Segít hogy ne legyen túlillesztés
        Dense(10, activation='softmax')
    ]
)

model_cnn.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model_cnn.summary()

history_cnn = model_cnn.fit(
    x_train, 
    y_train_cat,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

#Első Conv2D réteg szűrői 
first_conv_layer = next(layer for layer in model_cnn.layers if isinstance(layer, Conv2D))
filters, biases = first_conv_layer.get_weights()
print("Filters shape: ", filters.shape) #(3, 3, 1, 32) -> 32 szűrő 3x3, 1 channel

n_filters = 6 #mennyit mutatunk meg
fig, axs = plt.subplots(1, n_filters, figsize=(15,5))

for i in range(n_filters):
    f = filters[:, :, 0, i] #az i-edik szürő
    axs[i].imshow(f, cmap='gray')
    axs[i].axis('off')

#plt.show()

#Egy modell, ami az első conv réteget látja
layer_outputs = [layer.output for layer in model_cnn.layers[:2]] # A Conv + Pool
model_cnn.predict(x_test[:1])
activation_model = Model(inputs=model_cnn.inputs, outputs=layer_outputs)

#Egy kép a tesztből
img = x_test[0].reshape(1,28,28,1)

activations = activation_model.predict(img)

first_layer_activation = activations[0]
print("First layer activation shape: ", first_layer_activation.shape) #(1,26,26,32)

#Plotoljunk pár feature map-et
n_features = 6
fig, axs = plt.subplots(1, n_features, figsize=(15,5))

for i in range(n_features):
    axs[i].imshow(first_layer_activation[0, :, :, i], cmap='viridis')
    axs[i].axis('off')

plt.show()