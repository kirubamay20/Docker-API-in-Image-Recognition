from keras.models import load_model
from PIL import Image
import numpy as np
from flasgger import swagger

from flask import Flask, request
app=Flask(__name__)

model = load_model('./model.h5')

@app.route('/predict_digit', method=['POST'])

def predict_digit():

  im= Image.open(request.files[''])
  im2arr=np.array(im).reshape((1,1,28,28))
  return np.argmax(model.predict(im2arr))

if __name__ =='__main__':
  app.run()

