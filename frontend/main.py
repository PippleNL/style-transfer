import requests
import streamlit as st
from PIL import Image
import time

STYLES = {
    "candy": "candy",
    "composition 6": "composition_vii",
    "feathers": "feathers",
    "la_muse": "la_muse",
    "mosaic": "mosaic",
    "starry night": "starry_night",
    "the scream": "the_scream",
    "the wave": "the_wave",
    "udnie": "udnie",
}

# https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Style transfer web app")

# displays a file uploader widget
image = st.file_uploader("Choose an image")

# displays the select widget for the styles
style = st.selectbox("Choose the style", [i for i in STYLES.keys()])

# displays a button
if st.button("Style Transfer"):
    if image is not None and style is not None:
        files = {"file": image.getvalue()}
        res = requests.post(f"http://backend:8080/{style}", files=files)
        img_path = res.json()
        image = Image.open(img_path.get("name"))
        st.image(image)

        displayed_styles = [style]
        displayed = 1
        total = len(STYLES)

        st.write("Generating other models...")

        while displayed < total:
            for style in STYLES:
                if style not in displayed_styles:
                    try:
                        path = f"{img_path.get('name').split('.')[0]}_{STYLES[style]}.jpg"
                        image = Image.open(path)
                        st.image(image, width=500)
                        time.sleep(1)
                        displayed += 1
                        displayed_styles.append(style)
                    except:
                        pass
