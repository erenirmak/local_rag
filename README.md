# Chat with Your Multiple PDFs on Your Local System

Using a language model, you can talk with your PDF files. This also lets your language model do less mistakes and hallucinations.

## Setup Database
I used PostgreSQL in this project because it is lightweight and robust.
You can download it from here: https://www.postgresql.org/download/

## Setup Vector Extension
PostgreSQL has an extension: pgvector. You can use it to store embeddings.
You can download it from here: https://github.com/pgvector/pgvector
Follow the instructions to install the extension.
Then, install the python package:
```
pip3 install pgvector
```

## Requirements
datasets==2.15.0
huggingface_hub==0.19.4
langchain==0.0.341
pgvector==0.2.4
psycopg2==2.9.9
SQLAlchemy==2.0.23
torch==2.1.1+cu118
transformers==4.35.2
bitsandbytes==0.41.2.post2 (bitsandbytes-windows)
peft==0.6.2
accelerate==0.24.1
optimum==1.14.1
auto-gptq==0.5.1+cu118
safetensors==0.4.0

### Other Requirements
PostgreSQL database
pgvector extension
CUDA 11.8

## Language Model
I used Zephyr-7b-beta in this project. It is state-of-the-art language model
developed by HuggingfaceH4 on top of the Mistral-7b model.
Because of extensive VRAM requirements of the non-quantized model, I used its 4-bit
quantized version.
You can find the model here: https://huggingface.co/TheBloke/zephyr-7B-beta-GPTQ

For generating embeddings, I used all-mpnet-base-v2. It is one of the most commonly
models.
You can find the model here: https://huggingface.co/sentence-transformers/all-mpnet-base-v2


## OS & Hardware
I am currently using Windows 10. If you are using different OS, download & install
the packages for your needs.
I have Nvidia GTX 1660Ti Max-Q Design 6 GB on my laptop and quantized model fits in
my GPU, takes up to 5.7 GB of VRAM. If you don't have enough VRAM, you can try
different model from TheBloke's quantized models that can run on CPU.

# Talk with Your PDFs
One of the main problems with LLMs is that they hallucinate. Retrieval Augmented
Generation (RAG) is an algorithm developed by Meta AI researchers. The main idea
behind RAG is providing ground truth information to the LLM and let it generate
its responses based on this information. The source of the information can be
documents, can be databases, datasets or the internet. The responses generated
by the language model anchored to the facts better and decreases its hallucination
rates. Also, same approach can be used in analyzing, summarizing or deriving
information using your documents.

Put your PDF files in a folder and talk with the model. I provided some example
academic articles. You can use this approach in analyzing and summarizing CVs,
news, etc. And that's it. :)

You can design UI with web frameworks like Flask and even dockerize it. Now, you
have your own local language model that always up. You don't have to share your
private documents with 3rd parties to get the summarization. You can even use
the model to write your homework articles. :) Shhh

Enjoy!
