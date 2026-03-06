import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

model = tf.keras.models.load_model("skin_cancer_model.h5")

classes = ['akiec','bcc','bkl','df','mel','nv','vasc']

st.title("AI Skin Cancer Detection")

file = st.file_uploader("Upload Skin Image")

if file:

    image = Image.open(file)
    st.image(image)

    img = np.array(image)

    img = cv2.resize(img,(224,224))
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    pred = model.predict(img)

    class_id = np.argmax(pred)

    result = classes[class_id]

    st.write("Prediction:",result)
