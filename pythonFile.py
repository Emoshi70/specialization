from langchain.document_loaders import WebBaseLoader

#add website data
URL = ["https://www.mapua.edu.ph/pages/academics/undergraduate/intramuros-campus/school-of-electrical-electronics-and-computer-engineering/bachelor-of-science-in-computer-engineering"
       
       ]

#load the data
data = WebBaseLoader(URL)
#extract the content
content = data.load()


from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=256,chunk_overlap=50)
chunking = text_splitter.split_documents(content)

from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings

import os
from getpass import getpass

# get your free access token from HuggingFace and paste it here
HF_token = getpass()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HF_token

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key = HF_token,model_name = "BAAI/bge-base-en-v1.5"
)


from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(chunking, embeddings)


retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k":3})
query = "What is Bachelor’s Degree in Computer Engineering?"
docs_rel = retriever.get_relevant_documents(query)
print(docs_rel)


prompt = f"""
<|system|>>
You are an AI Assistant that follows instructions extremely well.
</s>
<|user|>
{query}
</s>
<|assistant|>
"""


from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

model = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-alpha",
                       model_kwargs={"temperature":0.5,
                                     "max_new_tokens":512,
                                     "max_length":64
                                    })

qa = RetrievalQA.from_chain_type(llm=model,retriever=retriever,chain_type="stuff")
response = qa(prompt)
print(response['result'])
