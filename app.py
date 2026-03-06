import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# ---------------------------
# Load Model
# ---------------------------

model = tf.keras.models.load_model("skin_cancer_model.h5")

classes = ['akiec','bcc','bkl','df','mel','nv','vasc']

# ---------------------------
# Page Setup
# ---------------------------

st.set_page_config(
    page_title="AI Skin Cancer Detection",
    layout="wide"
)

st.title("🧬 AI Skin Cancer Detection Dashboard")

st.write("Upload a skin lesion image to analyze possible skin cancer.")

# ---------------------------
# PDF Report Generator
# ---------------------------

def generate_report(result, confidence, risk):

    buffer = io.BytesIO()

    c = canvas.Canvas(buffer, pagesize=letter)

    c.setFont("Helvetica-Bold",16)
    c.drawString(150,750,"Skin Cancer AI Diagnosis Report")

    c.setFont("Helvetica",12)

    c.drawString(100,700,f"Predicted Disease: {result}")
    c.drawString(100,670,f"Confidence Score: {confidence*100:.2f}%")
    c.drawString(100,640,f"Cancer Risk Level: {risk}")

    c.drawString(100,600,"Note:")
    c.drawString(100,580,"This AI system provides an automated analysis.")
    c.drawString(100,560,"Consult a dermatologist for professional diagnosis.")

    c.save()

    buffer.seek(0)

    return buffer

# ---------------------------
# Upload Image
# ---------------------------

uploaded_file = st.file_uploader("Upload Skin Image")

if uploaded_file:

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file)

    col1.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)

    img = cv2.resize(img,(224,224))

    img_norm = img/255.0

    img_array = np.expand_dims(img_norm,axis=0)

    # ---------------------------
    # Model Prediction
    # ---------------------------

    prediction = model.predict(img_array)

    class_id = np.argmax(prediction)

    confidence = np.max(prediction)

    result = classes[class_id]

    # ---------------------------
    # Prediction Card
    # ---------------------------

    col2.subheader("Prediction Result")

    col2.metric("Predicted Disease", result)

    col2.metric("Confidence", f"{confidence*100:.2f}%")

    # ---------------------------
    # Risk Indicator
    # ---------------------------

    if result in ["mel","bcc","akiec"]:

        if confidence > 0.80:
            risk = "HIGH"
            col2.error("🔴 Cancer Risk Level: HIGH")

        elif confidence > 0.50:
            risk = "MEDIUM"
            col2.warning("🟠 Cancer Risk Level: MEDIUM")

        else:
            risk = "LOW"
            col2.info("🟡 Cancer Risk Level: LOW")

    else:

        risk = "LOW"
        col2.success("🟢 Cancer Risk Level: LOW")

    # ---------------------------
    # Probability Chart
    # ---------------------------

    st.subheader("Prediction Probabilities")

    fig, ax = plt.subplots()

    ax.bar(classes, prediction[0])

    ax.set_ylabel("Probability")

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45)

    st.pyplot(fig)

    # ---------------------------
    # Heatmap Visualization
    # ---------------------------

    st.subheader("Heatmap Visualization")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img,0.6,heatmap,0.4,0)

    st.image(overlay, caption="Skin Lesion Heatmap", use_column_width=True)

    # ---------------------------
    # Download PDF Report
    # ---------------------------

    st.subheader("Download Medical Report")

    pdf = generate_report(result,confidence,risk)

    st.download_button(
        label="Download Diagnosis Report",
        data=pdf,
        file_name="skin_cancer_report.pdf",
        mime="application/pdf"
    )
