import streamlit as st
import numpy as np
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from time import sleep
import sys

gpu_to_use = 0
device = torch.device(f'cuda:{gpu_to_use}' if torch.cuda.is_available() else 'cpu')

with st.sidebar:
    data = st.file_uploader(label = "upload a csv file",type = 'csv')

if data is not None:
    df = pd.read_csv(data)
    st.dataframe(df.head(5))
    
 
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
    

#seaborn, matplotlib, and 
# 5) if not specified choose an appropriate plot
prompt = """You're a highly skilled Python coder known for your mastery in pandas, seaborn, matplotlib, plotly.
When it comes to visualizations, your expertise shines through, crafting aesthetically pleasing plots with captivating color schemes.

### Rules:
1) print values above the bars of bar plot.
2) the output should be phrased with respect to question.
3) Use seaborn or plotly along with matplotlib.
4) do not print values above the bar if it gets too congested.
5) only give the code without explanation and comments.
6)Always import necessary libraries and load the dataset.

# You should strictly follow the above rule

Here's the scoop on the "Salaries.csv" dataset, boasting 78 records featuring diverse columns:

|   | rank | discipline | phd | service |  sex  | salary |
|---|------|------------|-----|---------|-------|--------|
| 0 | Prof |     B      | 56  |   49    | Male  | 186960 |
| 1 | Prof |     A      | 12  |    6    | Male  |  93000 |
| 2 | Prof |     A      | 23  |   20    | Male  | 110515 |
| 3 | Prof |     A      | 40  |   31    | Male  | 131205 |
| 4 | Prof |     B      | 20  |   18    | Male  | 104800 |

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

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.code(message["content"])


if question := st.chat_input("What is up?"):
   
    st.chat_message("user").markdown(question)   
   
    st.session_state.messages.append({"role": "user", "content": question})
    response = prompt.format(question)
    out = gen(response).replace("```","") 
    
    with st.chat_message("assistant"):
        st.code(out)    
        if out.find('plt.show()')!=-1:
            out_modified = out + 'st.image(plt.gcf(),format="png")'            
            exec(out_modified)
        elif out.find('fig.show()')!=-1:
            out_modified = out + 'st.image(plt.gcf(),format="png")' 
            exec(out_modified)
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
                f.seek(0)    
                st.text(f.read())   
                f.truncate(0)  
  
    # st.write("append content")
    st.session_state.messages.append({"role": "assistant", "content": out})

st.session_state

