import os
import shutil
import pickle
import argparse

from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.vectorstores.faiss import FAISS
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document as LangChainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from modules.file import SimpleDirectoryReader

import dotenv
dotenv.load_dotenv()



PATH = r'C:\Users\stgeorge\Desktop\pdfs\Walking the Plant Path.pdf'

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


def summarize(texts, temperature=0, max_tokens=1024, chain_type="map_reduce"):
    llm = OpenAI(temperature=temperature, max_tokens=max_tokens)
    summary_chain = load_summarize_chain(llm, chain_type=chain_type)
    summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
    print("Running summarize document chain...")
    return summarize_document_chain.run(texts)


def summarize_from_prompt(docs, prompt, max_tokens=600):
    chain = load_summarize_chain(
        OpenAI(temperature=0, max_tokens=max_tokens), 
        chain_type="map_reduce",
        return_intermediate_steps=True, 
        map_prompt=prompt, combine_prompt=prompt
    )
    return chain({"input_documents": docs}, return_only_outputs=True)['output_text']


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


docs = split_docs


# TODO: Use vectorstore, (CREATE FAISS INDEX FIRST) as a way to ask questions about the documents, instead of using Map_reduce...
# TODO: 

# Create a prompt template for summary report (detailed)
prompt_template = """Write a 500 word summary report about the following text, first generate a 1-2 paragraph summary, then clearly delineate and explain the important concepts and ideas discussed in detail:
{text}
SUMMARY REPORT:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
texts = list(map(lambda x: x.page_content, docs))
summary = summarize(texts)
# summary = summarize_from_prompt(docs, PROMPT)

print("SUMMARY: {}".format(summary))

input("STOP!")

with open("youtube_summary.txt", "w") as f:
    f.writelines(summary)


def get_topics(docs):
    prompt_template = """Extract at the 5 key topics discussed in the following text and return as a list:
    {text}
    TEXT: "A string of text example"
    TOPICS LIST:
    - Topic 1
    - Topic 2
    - Topic 3
    - Topic 4
    - Topic 5
    TOPICS LIST:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    topics = summarize_from_prompt(docs, PROMPT)
    topics_list = topics.split("- ")

    if topics_list[0] == '\n': topics_list = topics_list[1:]
    topics_list = list(map(lambda x: x.split("\n")[0], topics_list))

    return [t for t in topics_list if t != ''] # Remove null strings

topics_list = get_topics(docs)
print("Topics: {}".format(topics_list))
with open("extracted_topics.txt", "w") as f:
    for topic in topics_list:
        f.write(topic + "\n")




def get_blog_post(topic,
                  k=4,
                  documents=None,
                  search_index=None,
                  max_tokens=512,
                  prompt_template=None):              

    if documents is not None and search_index is None:
        source_chunks = []
        splitter = CharacterTextSplitter(separator=". ", chunk_size=1024, chunk_overlap=0)
        for source in documents:
            for chunk in splitter.split_text(source.text):
                source_chunks.append(LangChainDocument(page_content=chunk))

        search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())

    elif documents is None and search_index is None:
        raise ValueError("Must pass at least `documents` or `search_index`")

    prompt_template = prompt_template or """Use the context below to write a 500 word blog post about the topic below:
        Context: {context}
        Topic: {topic}
        Blog post:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "topic"]
    )
    llm = OpenAI(temperature=0, max_tokens=max_tokens)
    chain = LLMChain(llm=llm, prompt=PROMPT)

    def generate_blog_post(topic):
        docs = search_index.similarity_search(topic, k=k)
        inputs = [{"context": doc.page_content, "topic": topic} for doc in docs]
        results = chain.apply(inputs)
        print("{} Blog:\n".format(topic), results)
        return results

    return generate_blog_post(topic)


def build_index_and_write_blogs(docs, topics_list, index_path='index.pkl', save=True):
    if index_path is not None and os.path.exists(index_path):
        with open(index_path, 'rb') as f:
            index = pickle.load(f)
    else:
        index = FAISS.from_documents(docs, OpenAIEmbeddings())
    
    if save:
        with open(index_path, 'wb') as f:
            pickle.dump(index, f)

    blogs = []
    for i, topic in enumerate(topics_list):
        print("Writing blog for topic: {}".format(topic))
        post = get_blog_post(topic, search_index=index)
        blogs.append(post[0]) # maybe unlist
        print("Completed!")

    for i, blog in enumerate(blogs):
        with open('blog_{}.txt'.format(i), 'w') as f:
            f.write(blog['text'])

    return blogs

# blogs = build_index_and_write_blogs(docs, topics_list)
# print("Done writing and saving blogs!")