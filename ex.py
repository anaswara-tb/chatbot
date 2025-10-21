import streamlit as st

# Display text
st.title("WELCOME TO MY CHATBOT....")
st.write("Hello, world!")

# Add widgets
question = st.text_input("HOW CAN I HELP YOU TODAY???:")
if question:
    st.write(f"Hello im too...!")

    st.balloons()
    
