# Chat with Your Multiple PDFs on Your Local System

Using a language model, you can talk with your PDF files. This also makes language model less mistakes and hallucinations.

## Setup Database
I used PostgreSQL in this project because it is lightweight and robust.
You can download it from here: https://www.postgresql.org/download/

## Setup Vector Extension
PostgreSQL has an extension: pgvector. You can use it to store embeddings.
You can download it from here: https://github.com/pgvector/pgvector
Follow the instructions to install the extension.

## Requirements
datasets==2.15.0
huggingface_hub==0.19.4
langchain==0.0.341
pgvector==0.2.4
SQLAlchemy==2.0.23
torch==2.1.1+cu118
transformers==4.35.2
