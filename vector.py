import chromadb
import requests
import ollama

from pypdf import PdfReader
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi


# -------------------------
# READ PDF
# -------------------------

def read_pdf(path):
    text = ""

    reader = PdfReader(path)

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -------------------------
# READ TXT
# -------------------------

def read_txt(path):

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


# -------------------------
# READ WEBSITE
# -------------------------

def read_website(url):

    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    return soup.get_text(separator=" ")


# -------------------------
# GET YOUTUBE VIDEO ID
# -------------------------

def get_video_id(url):

    return url.split("v=")[1]


# -------------------------
# READ YOUTUBE TRANSCRIPT
# -------------------------

def read_youtube(url):

    video_id = get_video_id(url)

    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    text = ""

    for item in transcript:
        text += item.text + " "

    return text


# -------------------------
# CHUNK TEXT
# -------------------------

def chunk_text(text, chunk_size=500, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


# -------------------------
# LOAD ALL DATA
# -------------------------

all_text = ""

print("Reading PDFs...")

all_text += read_pdf(
    "data/context engineering paper.pdf"
)

print("Reading TXT...")

all_text += read_txt(
    "data/context_engineering.txt"
)

print("Reading Links...")

with open(
    "data/links.txt",
    "r",
    encoding="utf-8"
) as file:

    links = file.readlines()

for link in links:

    link = link.strip()

    if not link:
        continue

    try:

        if "youtube" in link:

            print("Reading YouTube:", link)

            all_text += read_youtube(link)

        else:

            print("Reading Website:", link)

            all_text += read_website(link)

    except Exception as e:

        print("Error:", e)


# -------------------------
# CHUNKING
# -------------------------

print("Creating chunks...")

chunks = chunk_text(all_text)

print("Total chunks:", len(chunks))


# -------------------------
# EMBEDDING MODEL
# -------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -------------------------
# CHROMADB
# -------------------------

client = chromadb.PersistentClient(
    path="chroma_db"
)

try:
    client.delete_collection(
        "context_engineering"
    )
except:
    pass

collection = client.create_collection(
    "context_engineering"
)

print("Generating embeddings...")

for i, chunk in enumerate(chunks):
    response = ollama.embeddings(
        model="llama3.2",
        prompt=chunk
    )
    embedding = response["embedding"]
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[chunk]
    )

print("Done.")
print("Vector database created successfully.")