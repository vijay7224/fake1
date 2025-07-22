import streamlit as st
import joblib
from PIL import Image
vectorizer=joblib.load("vectorizer")
model=joblib.load("model")
st.markdown(f"# :blue[FAKE NEWS DETECTION MODEL]")
inputs=st.text_input("ENTER THE NEWS")
if st.button("SUBMIT"):
    y = vectorizer.transform([inputs]) 
    z=model.predict(y)
    if z[0]==1:
        ima=Image.open("dp.jpg")
        st.image(ima,width=400)
        st.success("THIS NEWS NOT A FAKE")
    else:
        ima=Image.open("d.jpg")
        st.image(ima,width=400)
        st.error("THIS NEWS IS FAKE")
