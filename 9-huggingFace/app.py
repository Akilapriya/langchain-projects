import sys
import streamlit as st

# st.write(f"Python Executable: {sys.executable}")
# st.write(f"Looking for packages in: {sys.path}")
import validators
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

import os


# streamlit app
st.set_page_config(
    page_title="Langchain: Summarize Text from YT or Website", page_icon="🦜"
)
st.title("🦜 Langchain:summarize Text from YT or Website")
st.subheader("Summarize URL")

# Groq API key and url to be summarized
with st.sidebar:
    hf_api_key = st.text_input("Hugging face Token ", value="", type="password")

# heck if the user has actually entered a key yet
if not hf_api_key:
    st.info("Please enter your Hugging face token to continue.")
    st.stop()  # Pauses execution until a key is provided


generic_url = st.text_input("URL", label_visibility="collapsed")

# repo_id = "meta-llama/Llama-3.2-1B-Instruct"
repo_id = "mistralai/mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="conversational",
    # Pass your captured sidebar key here
    huggingfacehub_api_token=hf_api_key,
    temperature=0.7,
    # You can also manually specify a provider if 'auto' fails
    # provider="novita",
)


chat_model = ChatHuggingFace(llm=llm)
prompt_template = """
Provide a summary of the following content in 300 words:
content:{text}
"""

prompt = PromptTemplate(input_variables=["text"], template=prompt_template)

if st.button("summarize the content from YT or Website"):
    # validate all iputs
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please provide the info to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid url. It can be a YT video url or Website url")
    else:
        try:
            with st.spinner("waiting..."):
                # loading the website or yt video dat
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url, add_video_info=False
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        },
                    )

                docs = loader.load()

                # chain for summarizion
                chain = load_summarize_chain(
                    chat_model, chain_type="stuff", prompt=prompt
                )

                output_summary = chain.invoke(docs)
                st.success(output_summary)
        except Exception as e:
            st.exception(e)
