import sys
import streamlit as st

# st.write(f"Python Executable: {sys.executable}")
# st.write(f"Looking for packages in: {sys.path}")
import validators
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

import os


# streamlit app
st.set_page_config(
    page_title="Langchain: Summarize Text from YT or Website", page_icon="🦜"
)
st.title("🦜 Langchain:summarize Text from YT or Website")
st.subheader("Summarize URL")

# Groq API key and url to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API key", value="", type="password")

# heck if the user has actually entered a key yet
if not groq_api_key:
    st.info("Please enter your Groq API key to continue.")
    st.stop()  # Pauses execution until a key is provided
else:
    # 3. Pass the captured string straight to ChatGroq
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=groq_api_key,
    )

generic_url = st.text_input("URL", label_visibility="collapsed")

llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
content:{text}
"""

prompt = PromptTemplate(input_variables=["text"], template=prompt_template)

if st.button("summarize the content from YT or Website"):
    # validate all iputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the info to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid url. It can be a YT video url or Website url")
    else:
        try:
            with st.spinner("waiting..."):
                # loading the website or yt video dat
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url, add_video_info=True
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
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.invoke(docs)
                st.success(output_summary)
        except Exception as e:
            st.exception(e)
