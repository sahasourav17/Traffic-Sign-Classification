from PIL import Image
import numpy as np
import streamlit as st
from keras.models import load_model

model = load_model('TrafficSignDetection.h5')
classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',
            3:'Speed limit (50km/h)',
            4:'Speed limit (60km/h)',
            5:'Speed limit (70km/h)',
            6:'Speed limit (80km/h)',
            7:'End of speed limit (80km/h)',
            8:'Speed limit (100km/h)',
            9:'Speed limit (120km/h)',
           10:'No passing',
           11:'No passing vehicles over 3.5 tons',
           12:'Right-of-way at intersection',
           13:'Priority road',
           14:'Yield',
           15:'Stop',
           16:'No vehicles',
           17:'Vehicle > 3.5 tons prohibited',
           18:'No entry',
           19:'General caution',
           20:'Dangerous curve left',
           21:'Dangerous curve right',
           22:'Double curve',
           23:'Bumpy road',
           24:'Slippery road',
           25:'Road narrows on the right',
           26:'Road work',
           27:'Traffic signals',
           28:'Pedestrians',
           29:'Children crossing',
           30:'Bicycles crossing',
           31:'Beware of ice/snow',
           32:'Wild animals crossing',
           33:'End speed + passing limits',
           34:'Turn right ahead',
           35:'Turn left ahead',
           36:'Ahead only',
           37:'Go straight or right',
           38:'Go straight or left',
           39:'Keep right',
           40:'Keep left',
           41:'Roundabout mandatory',
           42:'End of no passing',
           43:'End no passing vehicles > 3.5 tons'
}


st.markdown(
    """
    <style>
    .header-style {
        font-size:25px;
        font-family:sans-serif;
        position:absolute;
        text-align: center;
        color: 032131;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .font-style {
        font-size:20px;
        font-family:sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .footer-style {
        font-size: 15px;
        font-family: sans-serif;
        position: fixed;
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #262339;
        color: white;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.header('AI Project')
st.markdown(
    '<p class="header-style">Traffic Sign Classification with DL</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="footer-style"> @Team RTX-3070 </p>',
    unsafe_allow_html=True
)

def load_image(img):
    image = Image.open(img)
    image = image.resize((30,30))
    image = np.expand_dims(image,axis = 0)
    image = np.array(image)
    return image

def classify(img):
    pred = model.predict_classes([img])[0]
    sign = classes[pred+1]

    column_1, column_2 = st.beta_columns(2)
    column_1.write("Traffic Sign : ")
    column_2.write(f"{sign}")

uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png','jpeg'],accept_multiple_files = False)


if uploadFile is not None:
    img = load_image(uploadFile)
    st.image(img)
    st.write("Image Uploaded Successfully")
    if st.button('classify image'):
        classify(img)

