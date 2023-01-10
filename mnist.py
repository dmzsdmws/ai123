import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np

@st.cache(allow_output_mutation=True)
def load():
    return load_model('cnn.h5')
model = load()
mode = True
st.write('MNISTを用いた手書き数字識別アプリ')
CANVAS_SIZE = 192
if st.button("書き"):
    mode = True
if st.button("消し"):
    mode = False
canvas = st_canvas(
        fill_color='#000000',
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode="freedraw" if mode else "transform",
        key='canvas'
    )

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Model Input')
    st.image(rescaled)

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val = model.predict(test_x.reshape(1, 28, 28))
    st.write(f'数字は: {np.argmax(val[0])}')
    st.bar_chart(val[0])
