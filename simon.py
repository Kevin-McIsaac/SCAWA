import pinecone 

from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import BaseCallbackHandler
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain.vectorstores import Pinecone

import streamlit as st

from typing import Any
import os

from dotenv import load_dotenv
load_dotenv() #get API keys

PROMPT = '''You are a helpful Strata legal expert in Western Australia answering questions about the "Strata Titles Act 1985" from a lot owner.

Start the answer with "An owner should always refer to their bylaws and strata plan in conjenction with the legislation".

Provide a detailed answer using the information from the legislation provided below. List relevant sections of the act. 

Do not make up answers. If you do not know say "I do not know".

    {context}

Question: {question}'''

class SimpleStreamlitCallbackHandler(BaseCallbackHandler):
    """ Copied only streaming part from StreamlitCallbackHandler """
    
    def __init__(self) -> None:
        self.tokens_area = st.empty()
        self.tokens_stream = ""
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.tokens_stream += token
        self.tokens_area.markdown(self.tokens_stream)

st.markdown('<style>[data-testid="stForm"] {border: 0px}</style>', unsafe_allow_html=True) # no form border 

@st.cache_resource
def makeBot(prompt_template, k, streaming, namespace):
    '''Set up connection to db and create the Qretrieval object
       only needs to be run once.'''

    pinecone.init(
        api_key= os.environ.get('PINECONE_API_KEY') ,  # find at app.pinecone.io
        environment=os.environ.get('PINECONE_ENV')     # next to api key in console
    )

    db = Pinecone.from_existing_index(os.environ.get('INDEX'), OpenAIEmbeddings())
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":k, 'namespace': namespace})

    PROMPT = PromptTemplate(template=prompt_template, 
                            input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
                        llm=ChatOpenAI(temperature=0, streaming= (streaming == 'Yes')), #, uses 'gpt-3.5-turbo' which is cheaper and better 
                        chain_type="stuff", 
                        retriever=retriever, 
                        chain_type_kwargs={"prompt": PROMPT}, 
                        return_source_documents=True)
    return qa

st.title("Simon's Playground")
st.caption("Use the inputs in the sidebar to experiment with the playground settings")

with st.form("chat"):           
    with st.sidebar:
        
        k = st.number_input("How many Articles should Simon consult", value= 4, 
                            help="This determines the number of similar article chunks to use in the prompt. Increasing this adds to the knowledge available but increases the cost/time to answer the inquiry")

        streaming = st.radio("Stream answers?", ("Yes", "No"),horizontal=True, index=0,
                            help="Streaming shows the output as its generated like chatGPT but the cost and #tokens does not work. Turn off to see costs")
        
        
        instructions = st.text_area('Instructions for Simon to answer the inquiry', 
                                    value=PROMPT,
                                    height = 300, 
                                    help='This is prepended to the inquiry and provides instructions on how to answer the inqueiry. Try chaning this to get a better result.')


    qa=makeBot(instructions, k, streaming, 'SCA_H5') 
    inquiry = st.text_area(f"Hi, I'm Simon, the Strata Chatbot? What is your inquiry?",
                        value="")
    submitted = st.form_submit_button("Submit")


if submitted:
    with get_openai_callback() as costs:
        result = qa(inquiry, callbacks=[SimpleStreamlitCallbackHandler()]) 
        #streaming callback handler breaks cost reporting
        if streaming == "No": 
            st.write(result['result']) 
        st.caption(f":blue[Cost {costs.total_cost*100 : 0.2f} US cents. Tokens {costs.total_tokens :,} = {costs.prompt_tokens :,} + {costs.completion_tokens :,}]")

    for source in result["source_documents"]: 
        st.markdown(f"<details><summary>{source.metadata.get('title')} [{source.metadata.get('source')]}"
                    f"</summary>source.metadata.get('url')\n{source.page_content} </details>", 
                    unsafe_allow_html=True)
        

st.info(
    '''This webpage provides a basic understanding of strata title living. It is general information only 
    and is not legal advice on strata titles. You should refer to the legislation available on the WA government website

    To the extent permitted by law, we will in no way be liable to you or anyone else for any loss or damage,
    however caused (including through negligence), which may be directly or indirectly suffered in connection with use
    of this document.
    '''
)