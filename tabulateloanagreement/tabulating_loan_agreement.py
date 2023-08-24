from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import openai
import os

class TabulateLoanAgreement:
    def __init__(self, model_name, temperature=0.0):
        self.model_name = model_name
        self.temperature = temperature

        load_dotenv()
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv("OPENAI_API_BASE")

        self.initialize_pipeline()

    def initialize_pipeline(self):
        self.load_text()
        self.process_text()
        self.create_vectorstore()
        self.create_conversation_chain()

    def load_text(self):
        with open("loanagreement .txt", 'r', encoding='utf-8') as file:
            self.text = file.read()
            #print(self.text)

    def process_text(self):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.text_chunks = text_splitter.split_text(self.text)

    def create_vectorstore(self):
        embeddings = SentenceTransformerEmbeddings(model_name=self.model_name)
        self.vectorstore = FAISS.from_texts(texts=self.text_chunks, embedding=embeddings)

    def create_conversation_chain(self):
        llm = ChatOpenAI(temperature=self.temperature, model_kwargs={"engine": "GPT3-5"})
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=False)
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(),
            memory=memory
        )

    def query_conversation_chain(self, question):
        response = self.conversation_chain({'question': question})
        return response["answer"]

# Create an instance of the TabulateLoanAgreement class
pipeline = TabulateLoanAgreement(
    model_name="all-MiniLM-L6-v2",
    temperature=0.0
)

# Use the pipeline to query the conversation chain
response = pipeline.query_conversation_chain("Summarize the agreement by giving the start and end dates, interest rate, and loan amount. Present this information in a four-column table.")
print(response)
