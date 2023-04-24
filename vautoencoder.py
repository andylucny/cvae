from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K
import numpy as np
import cv2

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
#x_train = np.round(x_train)
#x_test = np.round(x_test)

for i in range(50):
    cv2.imwrite('mnist/inp'+str(i).zfill(5)+".png",np.asarray(x_train[i]*255,np.uint8))

inp = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
print(inp.shape)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(inp)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='sigmoid', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
print(x.shape)
latent_space = Flatten()(x)
print(latent_space.shape)
x = Reshape((4,4,8))(latent_space)
print(x.shape)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

out = x
print(out.shape)

autoencoder = Model(inp, out)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.summary()

autoencoder.fit(x_train, x_train,epochs=100,batch_size=128,shuffle=True,validation_data=(x_test, x_test))

scores = autoencoder.evaluate(x_test, x_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Save model and weights
model_name = 'keras_mnist_autoencoder_model.h5'
autoencoder.save(model_name)
print('Saved trained model at %s ' % model_name)

# use the model on few samples
input_images = x_train[0:10]
output_images = autoencoder.predict(input_images)
for i in range(10):
    cv2.imwrite('mnist/out'+str(i).zfill(5)+".png",np.asarray(output_images[i]*255,np.uint8))

# use the first half of autoencoder as an encoder
#encoder = Model(inp, latent_space)
#encoder.summary()
orig_input = Input(shape=(28,28,1), dtype=float)
x = orig_input
for layer in autoencoder.layers[:8]:
    x = layer(x)

encoder = Model(orig_input, x)
encoder.summary()

coded = encoder.predict(input_images)
print(coded[0])

# use the second half of autoencoder as a decoder
code_shape = autoencoder.layers[8].get_input_shape_at(0)
print(code_shape)

coded_input = Input(shape=(128,), dtype=float)
x = coded_input
for layer in autoencoder.layers[8:]:
    x = layer(x)

decoder = Model(coded_input, x)
decoder.summary()

decoded = decoder.predict(coded)
print(decoded.shape)
cv2.imwrite('decoded'+str(0).zfill(5)+".png",np.asarray(decoded[0]*255,np.uint8))
coded[0,15] /= 2
decoded = decoder.predict(coded)
cv2.imwrite('deformed'+str(0).zfill(5)+".png",np.asarray(decoded[0]*255,np.uint8))

