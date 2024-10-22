### Importing library and modules
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import torch
import os

### Title of App
st.title("Welcome to VJ's Amzing GPT App")

### Enter BUID
BUID = U36258376

### OpenAI Secret Key
my_secret_key = st.secrets['MyOpenAIKey']

### Load GPT2 from hugging face
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

### Prompt input on streamlit app by user
prompt = st.text_input("Hello, please enter your prompt below and see the wonder happening", "Enter your prompt please")
max_tokens = st.number_input("Enter the number of tokens for the response", min_value=1, max_value=500, value=100)

### Engineer the types of responses based on temperature
if st.button("Generate Response"):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs_high = model.generate(
        inputs['input_ids'], 
        max_length=max_tokens,
        do_sample=True, 
        temperature=1.6
    )
    creative_response = tokenizer.decode(outputs_high[0], skip_special_tokens=True)
    st.write("Creative Response:")
    st.write(creative_response)

    outputs_low = model.generate(
        inputs['input_ids'], 
        max_length=max_tokens, 
        do_sample=True, 
        temperature=0.2
     )
    predictable_response = tokenizer.decode(outputs_low[0], skip_special_tokens=True)
    st.write("Predictable Response:")
    st.write(predictable_response)
