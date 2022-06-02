import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

from capture_cam import *


@st.cache
def load_model(model_name):
    detector = cam(model_name=model_name)

    return detector

@st.cache
def load_image(img_path):
    """load"""
    im = Image.open(img_path)
    return im

class VideoProcessor(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="rgb24")
        
        print("\n ========== \n", type(img), "\n ========== \n")
        
        img = detector.image_detection(frame)

        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    '''
    Fire detection app.
    '''
    global detector

    detector = load_model("./model/best.pt")

    st.title('Fire Detection')
    st.sidebar.title('Fire Detection')

    activities = ['Image detection', 'Video detection', 'Webcam detection']
    choice = st.sidebar.selectbox("Select activity : ", activities)

    if choice == 'Image detection':
        st.subheader("Detection on Image")

        st.text("Upload an image to detect fire")
        image_file = st.file_uploader("Choisir une image", type=["jpg", "png", "jpeg"])
        
        st.text("Or paste an image url")
        url_image = st.text_input(label="Image URL", value="")

        if image_file is not None:

            # Convert the file to an opencv image.
            # file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            # opencv_image = cv2.imdecode(file_bytes, 1)
            # print("\n ========= \n",opencv_image,"\n ========= \n")

            if st.button("Start"):
                original_image = load_image(image_file)
                st.text("Fire detection on image")
                img = detector.image_detection(original_image)

                st.image(img, use_column_width=True)

            else:
                original_image = load_image(image_file)

                st.text("Image loaded")
                st.image(original_image, use_column_width=True)

                # st.image(opencv_image, channels="BGR")

        if url_image != "":

            response = requests.get(url_image)

            if st.button("Start"):
                # original_image = load_image(BytesIO(response.content))
                st.text("Fire detection on image")
                img = detector.image_detection(url_image)

                st.image(img, use_column_width=True)

            else:
                original_image = load_image(BytesIO(response.content))

                st.text("Image loaded")
                st.image(original_image, use_column_width=True)

    elif choice == 'Video detection':
        pass

    elif choice == 'Webcam detection':
        webrtc_streamer(key="example", video_processor_factory=VideoProcessor,rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

if __name__ == "__main__":
    main()