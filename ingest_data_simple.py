import pickle
import argparse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings

import dotenv
dotenv.load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True, type=str, help='Path to data directory')
parser.add_argument('--index_path', default="vectorstore.pkl", type=str, help='Path to save index')
parser.add_argument('--chunk_size', default=1000, type=int)
parser.add_argument('--chunk_overlap', default=50, type=int)
args = parser.parse_args()


# Load data
from modules.file import SimpleDirectoryReader
loader = SimpleDirectoryReader(args.data_path, recursive=True, exclude_hidden=True)
documents = loader.load_data()

# Convert to langchain format
docs = []
for doc in documents:
    docs.append(doc.to_langchain_format())

doc_lengths = lambda x: list(map(lambda x: len(x.page_content), x))
print(doc_lengths(docs))

print("Max document length (before chunking): {}".format(max(doc_lengths(docs))))

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
)
split_docs = text_splitter.split_documents(docs)
print("Max split document length (after chunking): {}".format(max(doc_lengths(split_docs))))

# Load Data to vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embeddings)

# Save the vectorstore
with open(args.index_path, 'wb') as f:
    pickle.dump(vectorstore, f)
