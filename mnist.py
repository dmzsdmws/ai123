import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np


def load():
    return load_model('cnn.h5')
model = load()

st.title('MNISTを用いた手書き数字識別アプリ')
st.write('# ↓で書いてください')
CANVAS_SIZE = 192

col1, col2 = st.columns(2)
mode = !st.checkbox("消しますか", True)
    
if mode == True:
   with col1:
    canvas = st_canvas(
        fill_color='#000000',
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode="freedraw",
        key='canvas'
    )
if mode == False:
   with col1:
    canvas = st_canvas(
        fill_color='#000000',
        stroke_width=100,
        stroke_color='#000000',
        background_color='#000000',
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode="freedraw",
        key='canvas'
    )

if canvas.image_data is not None:
    img = canvas.image_data.astype(np.uint8)
    img = cv2.resize(img, dsize=(28, 28))
    preview_img = cv2.resize(img, dsize=(CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_NEAREST)

    col2.image(preview_img)

    x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = x.reshape((-1, 28, 28, 1))
    y = model.predict(x).squeeze()

    st.write('## 結果は: %d' % np.argmax(y))
    st.bar_chart(y)
