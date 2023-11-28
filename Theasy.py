


import os # file system
from huggingface_hub import login
# get model & pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
# load & process documents
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# get embeddings
from langchain.embeddings import HuggingFaceEmbeddings
# save embeddings to vector database
from langchain.vectorstores.pgvector import PGVector
# RAG chain
from langchain.chains import RetrievalQA

# login Huggingface
HUGGINGFACEHUB_API_TOKEN = "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"
login(HUGGINGFACEHUB_API_TOKEN)

# Load Model from Huggingface
model_name = "TheBloke/zephyr-7B-beta-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Create Pipeline
pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=5000,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
)
    
# LangChain LLM definition based on pipeline
llm = HuggingFacePipeline(pipeline=pipe) # [tokenizer -> model] linked

# Load PDF Files
pdf_path = "articles"
file_list = os.listdir(pdf_path)
os.chdir(pdf_path)
all_contents = []
for file in file_list:
    file_loader = PyPDFLoader(file)
    file_contents = file_loader.load() # returns splitted "list" of Document objects
    all_contents.append(file_contents)

# Rearrange Document objects
documents = []
for doc_list in all_contents:
    if type(doc_list) == list:
        for doc in doc_list:
            documents.append(doc)
    else:
        documents.append(doc_list)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

# Embedding setup
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs) # sentence transformer embeddings

# Vectordb Setup
connection_string = PGVector.connection_string_from_db_params(driver="psycopg2",
                                                              user="postgres",
                                                              password="e1r2e3n4",
                                                              host="localhost",
                                                              port="5432",
                                                              database="vectordb")

vectordb = PGVector.from_documents(documents=all_splits,
                                   embedding=embeddings,
                                   collection_name = "articles",
                                   connection_string=connection_string,
                                   pre_delete_collection=True)

retriever = vectordb.as_retriever() # sentence transformer embeddings -> vectordb -> retriever (converted)

#######################
qa = RetrievalQA.from_chain_type(
    llm=llm, # zephyr model
    chain_type="stuff", # refine, map_reduce, map_rerank
    retriever=retriever, # sentence transformer in the background
    verbose=True
)

def run_my_rag(qa, query):
    result = qa.run(query)
    print("\nResult: ", result)

print("Welcome to chat interface! Please type your query:")


# RAG experiment:
while 1:
    query = input("Custom query: ")
    if query == "exit":
        break
    
    run_my_rag(qa, query)





