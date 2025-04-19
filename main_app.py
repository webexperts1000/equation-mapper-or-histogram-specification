import streamlit as st
from histogram_tool import histogram_tool_ui
from equation_tool import equation_tool_ui

st.set_page_config(page_title="Image Toolbox", layout="wide")
st.title("🧰 Digital Image Processing")

# Sidebar navigation
tool = st.sidebar.radio("Choose a tool:", ["🧮 Equation Mapper", "📊 Histogram Specification"])

# Conditional tool loading
if tool == "📊 Histogram Specification":
    histogram_tool_ui()
elif tool == "🧮 Equation Mapper":
    equation_tool_ui()
