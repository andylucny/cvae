from keras.datasets import mnist
from keras.layers import Input
from keras.models import Model
from keras.models import load_model
import numpy as np
import cv2

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

x_train = preprocess_images(x_train)
x_test = preprocess_images(x_test)

model_name = 'encoder229.h5'
encoder = load_model(model_name)

encoder.summary()

batch_size = 512
y_codes = np.zeros((len(x_train),2),np.float32)    
for i in range(0,len(x_train),batch_size):
    a, b = i, min(len(x_train),i+batch_size)
    input_images = x_train[a:b] 
    output_codes = encoder.predict(input_images)
    y_codes[a:b] = output_codes[:,:2]

colors = { 0 : (0,0,255), 1 : (0,255,255), 2 : (0,255,0), 3 : (255,255,0), 4 : (255,255,255), 5: (160,160,160), 6: (255,0,0), 7: (255,0,255), 8: (80,80,0), 9 : (80,0,0) }

def display(points,types):
    ext = 1.6448536
    points = (points+ext)/(2*ext)
    v = 800
    graph = np.zeros((v,v,3),np.uint8)
    for i in range(len(points)):
        cv2.circle(graph,(int(v*points[i,0]),int(v*points[i,1])),2,colors[types[i]],cv2.FILLED)
    for j in range(10):
        cv2.rectangle(graph,(j*32,0),((j+1)*32,32),colors[j],cv2.FILLED)
        cv2.putText(graph,str(j),(j*32+8,32-8),0,0.9,(0,0,0))
    return graph

points = y_codes

graph = display(points,y_train)
cv2.imwrite('latent-space.png',graph)
cv2.imshow('points',graph)
cv2.waitKey(0)
cv2.destroyAllWindows()
