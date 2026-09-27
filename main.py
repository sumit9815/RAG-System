# code of main.py
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# * Initializing Embedding Model
embedding_model= HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# * Load the existing Chroma vector database from disk properly
vectorstore = Chroma(persist_directory="Chroma-VDB", embedding_function=embedding_model)

# * Initializing the retriever
retriever = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
)

llm_endpoint = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1")
llm = ChatHuggingFace(llm=llm_endpoint)

# * Creating Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.
Use only the provided context to answer the question.
If the answer is not present in the context, say: 'I could not find the answer in the document'""",
        ),
        (
            "human",
            """Context: {context}
Question: {question}""",
        ),
    ]
)

print("RAG system is ready!")
print("Press '0' to exit.")

while True:
    query = input("\nYou : ")
    if query.strip() == "0":
        print("Exiting RAG system. Goodbye!")
        break

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({"context": context, "question": query})

    response = llm.invoke(final_prompt)
    print(f"\nAI : {response.content}")
