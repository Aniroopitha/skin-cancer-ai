import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Load model
model = tf.keras.models.load_model("skin_cancer_model.h5")

classes = ['akiec','bcc','bkl','df','mel','nv','vasc']

# Page config
st.set_page_config(
    page_title="AI Skin Cancer Detection",
    layout="wide"
)

st.title("🧬 AI Skin Cancer Detection Dashboard")

st.write("Upload a dermatoscopic skin image to analyze possible skin cancer.")

# -----------------------
# GradCAM Function
# -----------------------

def get_gradcam(model, img_array, last_conv_layer_name="Conv_1"):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        class_idx = tf.argmax(predictions[0]).numpy()

        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap,0)

    heatmap = heatmap / np.max(heatmap)

    return heatmap


# -----------------------
# Upload Image
# -----------------------

uploaded_file = st.file_uploader("Upload Skin Image")

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    col1.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)

    img = cv2.resize(img,(224,224))

    img_norm = img/255.0

    img_array = np.expand_dims(img_norm,axis=0)

    prediction = model.predict(img_array)

    class_id = np.argmax(prediction)

    confidence = np.max(prediction)

    result = classes[class_id]

    # -----------------------
    # Prediction Card
    # -----------------------

    col2.subheader("Prediction Result")

    col2.metric("Predicted Disease", result)

    col2.metric("Confidence", f"{confidence*100:.2f}%")

    if result in ["mel","bcc","akiec"]:
        col2.error("⚠ Possible Skin Cancer Detected")
    else:
        col2.success("✅ Benign Skin Lesion")


    # -----------------------
    # Probability Chart
    # -----------------------

    st.subheader("Prediction Probabilities")

    fig, ax = plt.subplots()

    ax.bar(classes, prediction[0])

    ax.set_ylabel("Probability")

    ax.set_xticklabels(classes, rotation=45)

    st.pyplot(fig)


    # -----------------------
    # GradCAM Heatmap
    # -----------------------

    heatmap = get_gradcam(model,img_array)

    heatmap = cv2.resize(heatmap,(224,224))

    heatmap = np.uint8(255*heatmap)

    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

    superimposed = heatmap * 0.4 + img

    st.subheader("GradCAM Visualization")

    st.image(superimposed.astype("uint8"), caption="Highlighted Lesion Region")
