import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions


load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name="text-embedding-3-small"
)

chroma_client = chromadb.PersistentClient(path="chromadb_data")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
) 

client = OpenAI(api_key=openai_key)

resp = client.chat.completions.create(
    ##model="gpt-3.5-turbo",
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a Helping hand."},
        {"role": "user", "content": "What is human life expectancy in the United Kingdom ?"}
    ],
)

##useful to debug output

##print(resp.choices[0].message.content)

def load_documents_from_directory(directory_path):
    print("=== Loading documents from directory ===")
    
    documents = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename),
                "r", 
                encoding="utf-8"
            ) as file:
                documents.append({
                    "id": filename,
                    "text": file.read()
                    })
                
    return documents
    
    
#function to split text into chunks
    
def split_text(text, chunk_size=1000, chunk_overlap=20):
        chunks=[]
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - chunk_overlap
            
        return chunks
    
directory_path = "./news-articles"
documents = load_documents_from_directory(directory_path)
    
print(f"Loaded {len(documents)} documents")

#Split documents into chunks
chunked_documents = []

for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({
            "id": f"{doc['id']}_chunk_{i+1}",
            "text": chunk
        })
    
    
##  Uncheck to see chunked docs    
##  print(f"Split documents into chunks : {len(chunked_documents)}")

## function to generate embeddings using OpenAI APi
def get_openai_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    print("==== Generating embeddings... ====")
    return embedding

#Generate embeddings for the document chunks
for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_openai_embedding(doc["text"])

### print(doc["embedding"])

#Upsert documents with embeddings into ChromaDB

for doc in chunked_documents:
    print("==== Inserting chunks into db ====")
    collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
    )
    
# Funtion to query documents
def query_documents(question, n_results=2):
    #query_embedding = get_openai+embedding(question)
    results =collection.query(query_texts=question, n_results=n_results)
    
    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks

    # for idx, document in enumerate(results["documents"](0))
    #     doc_id = results["ids"][0][idx]
    #     distance = results["distances"][0][idx]
    #     print(f"Found document chunk: {document}")


def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question answering tasks. Use the following pieces of "
        "retrived context to answerthe questions. If you do not know the answer, say that you"
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion" + question
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
         {
            "role": "systems",
            "context": prompt,
         },
            {
            "role": "user",
            "content": question,
            }
    ]
)
    
    answer = response.choices[0].message
    return answer

question = "tell me about Getting Admission in an English Taught German Univeristy ?"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer)