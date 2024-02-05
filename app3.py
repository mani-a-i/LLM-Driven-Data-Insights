import streamlit as st
import numpy as np
import time
import torch
import pandas as pd
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from time import sleep
import json
import plotly.graph_objs as go
import sys
import os

gpu_to_use = 0
device = torch.device(f'cuda:{gpu_to_use}' if torch.cuda.is_available() else 'cpu')
 
@st.cache_resource  
def deepseek():    
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-7b-instruct-v1.5",load_in_4bit = True)
    return model

@st.cache_resource 
def deepseek_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-7b-instruct-v1.5",load_in_4bit = True)
    return tokenizer

model = deepseek()
tokenizer = deepseek_tokenizer()

def gen(prompt):
    toks = tokenizer(prompt,return_tensors = 'pt').to(device)
    output = model.generate(
        **toks,
        top_k = 1,
        do_sample = True,
        temperature = 0.2,
        max_new_tokens = 300,
        eos_token_id = [10897,63,4686,10252],
    )
    output = output[0][len(toks.input_ids[0]):]
    output = tokenizer.decode(output)
    return output  
    


prompt = """You're a highly skilled Python coder known for your mastery in pandas, seaborn, matplotlib, plotly.
When it comes to visualizations, your expertise shines through, crafting aesthetically pleasing plots with captivating color schemes.

### Rules:
1) print values above the bars of bar plot.
2) the output should be phrased with respect to question.
3) Use seaborn or plotly along with matplotlib.
4) do not print values above the bar if it gets too congested.
5) only give the code without explanation and comments.
6) Always import necessary libraries and load the dataset.
7) Plot only when told.
8) Show does not always means you have to plot 
    # Follow the example below
    User: Show me columns of the dataset
    Assistant:
    import pandas as pd
    df = pd.read_csv("Dataset.csv")
    print(df.columns)

# You should strictly follow the above rule

Here's the scoop on the dataset "{}" boasting records featuring diverse columns:
{}

# Follow the example below
User: Give me salary distribution across all sex
Assistant: 
import pandas as pd
df = pd.read_csv("Salaries.csv")
salary_distribution = df.groupby('sex')['salary'].mean()
colors = ['skyblue', 'lightcoral']
salary_distribution.plot(kind='bar', color=colors)
plt.title('Salary Distribution Across Sexes')
plt.xlabel('Sex')
plt.ylabel('Average Salary')
plt.show()

# Assist user

User: {}
Assistant:
```python
"""

def img_array(image):  
   
    img = Image.open(image)
    img_array = np.array(img)
    return img_array
    
def load_json(path):      
    with open(path, 'r') as file:        
        json_data = json.load(file)
    return json_data
    
with st.sidebar as sidebar:
    data = st.file_uploader(label = "upload a csv file",type = 'csv') 
    
if data is not None: 
    file_path = os.path.join(os.getcwd(), data.name)
    with open(file_path, "wb") as f:
        f.write(data.getbuffer())
    st.sidebar.write("File {} uploaded successfully".format(data.name))
    df = pd.read_csv(data) 
    st.sidebar.dataframe(df.head(5))
    prompt = prompt.format(data.name,df.sample(5),{})
    print("prompt after upload:{}".format(prompt)) 

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], np.ndarray):
            img = Image.fromarray(np.uint8(message['content']))
            st.code(message["code"])
            st.image(img)
            
        elif isinstance(message["content"], dict):
            plot = go.Figure(load_json('./output_figure.json'))
            st.code(message["code"])
            st.plotly_chart(plot)
            
        else:
            st.code(message["code"])
            st.text(message["content"])


if question := st.chat_input("What is up?"):
   
    st.chat_message("user").markdown(question)   
   
    st.session_state.messages.append({"role": "user","code": "", "content": question})
    response = prompt.format(question)
    print("prompt after question:{}".format(response))
    out = gen(response).replace("```","") 
    
    with st.chat_message("assistant"):
        st.code(out) 
        
        if out.find('plt.show()')!=-1:            
            out_modified = out + '''\nplt.savefig('output_image.png')'''+'''\nst.image('output_image.png')'''
            exec(out_modified)
            st.session_state.messages.append({"role": "assistant","code":out, "content": img_array("output_image.png")}) 
            
        elif out.find('fig.show()')!=-1:
            out_modified = out+'''\nimport plotly.io as pio''' +'''\npio.write_json(fig,'output_figure.json')'''+'''\nst.plotly_chart(fig)'''             
            exec(out_modified)
            st.session_state.messages.append({"role": "assistant","code":out ,"content": load_json('./output_figure.json')})
            
        else: 
            @contextmanager
            def custom_redirect(file_obj):
                org = sys.stdout
                sys.stdout = file_obj
                try:
                    yield file_obj
                finally:
                    sys.stdout = org
                    
            with open('./log.txt','r+') as f:
                with custom_redirect(f):
                    exec(out) 

            with open('./log.txt','r+') as f:
                f.seek(0)            
                content = f.read()
                st.text(content)
                st.session_state.messages.append({"role": "assistant","code":out,"content": content})
                f.truncate(0)  

# st.session_state
