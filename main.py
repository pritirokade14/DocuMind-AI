from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
     persist_directory="chroma_db",
     embedding_function=embedding_model
)

retriever = vector_store.as_retriever(
    search_type = 'mmr',
    search_kwargs = {
        'k': 4,
        'fetch_k': 10,
        'lambda_mult': 0.5
    }
)

# Configured with Groq and openai/gpt-oss-20b
llm = ChatGroq(model="openai/gpt-oss-20b")

prompt = ChatPromptTemplate.from_messages([
    ('system',
     """
You are a helpful AI assistant.
Use only the provided context to answer the question.
If the answer is not present in the context,
say: "I could not find the answer in the document."
"""),
    ('human',
     """
context: {context}
question: {question}
""")
])

print("RAG system is created")
print("Enter 0 to exit")

while True:
    query = input("You : ")
    if query == "0":
        break

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs]) 

    final_prompt = prompt.invoke({
        "context": context,
        "question": query 
    })

    response = llm.invoke(final_prompt)
    print(f"\n AI : {response.content}")