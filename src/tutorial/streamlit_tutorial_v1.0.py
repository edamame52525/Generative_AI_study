import streamlit as st

st.title('Hello, Streamlit!')
st.write('これはシンプルなアプリです。')

# スライダーで数値を選択
x = st.slider('xを選んでください', 0, 100)
st.write('選択した値:', x)

# インタラクティブなプロット
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot(np.sin(np.linspace(0, x / 10, 100)))
st.pyplot(fig)
