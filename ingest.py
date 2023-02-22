import pickle
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings

import os
import shutil
from modules.file import SimpleDirectoryReader

import dotenv
dotenv.load_dotenv()

PATH = r'C:\Users\stgeorge\Desktop\BlackRiverGPT\DocsGPT\data\md_files\pdfs\Walking_the_Plant_Path.pdf'

parser = argparse.ArgumentParser()
parser.add_argument('-p',  '--path', default=PATH, type=str, help='Path to single data file or directory of text-based files')
parser.add_argument('-ip', '--index_path', default="vectorstore.pkl", type=str, help='Path to save index')
parser.add_argument('-cs', '--chunk_size', default=1000, type=int)
parser.add_argument('-co', '--chunk_overlap', default=50, type=int)
parser.add_argument('-s',  '--separator', default=" ", type=str)
parser.add_argument('-d' , '--debug', action='store_true', default=False)
parser.add_argument('-t' , '--remove_tabs', action='store_true', default=True)
parser.add_argument('-r' , '--recursive_text_splitter', action='store_true', default=False)
args = parser.parse_args()


# Guesstimate cost
def estimate_cost(docs, char_per_penny=126556):
    c = 0
    for d in docs: c += len(d.page_content)
    cost = (c / char_per_penny) # in cents
    dollars = int(cost / 100)
    cents = str(cost).split(".")[-1][:2]
    print("Estimated cost ($): " + str(dollars) + "." + cents)
    return cost


## HACKY WORKAROUND
def hack_add_dir_and_cleanup(path):
    if os.path.isdir(path):
        return path

    # Copy to new tempdir for SimpleDirectoryReader (fuck UnstructuredTextReader)
    new_dir = copy_to_temp_dir(path)

    # Load single document
    loader = SimpleDirectoryReader(new_dir, recursive=True, exclude_hidden=True)
    documents = loader.load_data()

    # Cleanup
    shutil.rmtree(new_dir)
    return documents


def copy_to_temp_dir(path):
    # Get the base directory
    base_dir = os.path.dirname(path)

    # Create a new directory name
    new_dir = os.path.join(
        base_dir, os.path.basename(path).split(os.extsep)[0]
    )
    # Create the new directory
    os.mkdir(new_dir)

    # Copy the file to the new directory
    shutil.copy(path, new_dir)

    # Return the new directory path
    return new_dir


# Load Data
assert os.path.exists(args.path), "Filepath {} doesn't exist!".format(args.path)

if os.path.isdir(args.path):
    # Load directory and parse documents
    loader = SimpleDirectoryReader(args.path, recursive=True, exclude_hidden=True)
    documents = loader.load_data()
else:
    # Load and parse Single file
    documents = hack_add_dir_and_cleanup(args.path)

if args.debug:
    documents = documents[:2]

# Convert to langchain format
docs = []
for doc in documents:
    docs.append(doc.to_langchain_format())

# For Some PDFs that are a PITA with \t everywhere...
if args.remove_tabs:
    docs[0].page_content = docs[0].page_content.replace("\t", " ")

# Get lengths
doc_lengths = lambda x: list(map(lambda x: len(x.page_content), x))
print(doc_lengths(docs))
print("Max document length (before chunking): {}".format(max(doc_lengths(docs))))

# Split text
if args.recursive_text_splitter:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
else:
    text_splitter = CharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, separator=args.separator
    )

split_docs = text_splitter.split_documents(docs)
print("Max split document length (after chunking): {}".format(max(doc_lengths(split_docs))))
print("# Split docs:", len(split_docs))

est_cost = estimate_cost(split_docs)
inp = input("Acceptable to proceed? [y/n]\n> ")

if inp.lower() in ['y', 'yes', 'yeet']:
    # Load Data to vectorstore
    print("Creating vectorstore from split documents and OpenAI embeddings...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Save the vectorstore
    with open(args.index_path, 'wb') as f:
        pickle.dump(vectorstore, f)
else:
    print("No action taken. No $ spent.")
