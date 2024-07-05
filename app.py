import gradio as gr


import os
hftoken = os.environ["hftoken"]

from langchain_huggingface import HuggingFaceEndpoint

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
# repo_id = "google/gemma-2-9b-it"
# repo_id = "meta-llama/Meta-Llama-3-8B-Instruct" # answers the question well, but continues the text and does not stop when its necessary. often ends in incomplete responses.
# repo_id = "mistralai/Mixtral-8x22B-Instruct-v0.1"
llm = HuggingFaceEndpoint(repo_id = repo_id, max_new_tokens = 256, temperature = 0.7, huggingfacehub_api_token = hftoken, top_p=0.9)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.csv_loader import CSVLoader


loader = CSVLoader(file_path='aiotsmartlabs_faq.csv', source_column = 'prompt')
data = loader.load()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings

# CHECK MTEB LEADERBOARD & FIND BEST EMBEDDING MODEL
model = "BAAI/bge-m3"
model = "BAAI/bge-large-en-v1.5"
embeddings = HuggingFaceEndpointEmbeddings()
# embeddings = HuggingFaceEmbeddings(model = model)

vectorstore = Chroma.from_documents(documents = data, embedding = embeddings)
retriever = vectorstore.as_retriever()

# from langchain.prompts import PromptTemplate

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""Given the following context and a question, generate a complete and detailed answer with finished sentences based on the provided context only.

In your answer, try to use as much text as possible from the "response" section in the source document context without making significant changes.
If someone asks "Who are you?" or a similar question, reply with "My name is Chitti, a chatbot. I'm Rishi's assistant built using a Large Language Model!"
If the answer is not found in the context, kindly state "I don't know. Please ask Rishi on Discord at https://discord.gg/6ezpZGeCcM or email rishi@aiotsmartlabs.com." Do not attempt to make up an answer.

CONTEXT: {context}

QUESTION: {question}""")

# prompt = ChatPromptTemplate.from_template("""As an AI assistant for AIoT SMART Labs, your task is to provide accurate answers based on the given context.

# 1. **Use the context:** Generate an answer based only on the context provided. Try to use as much text as possible from the "response" section in the source document without making significant changes.
# 2. **Identify yourself:** If someone asks "Who are you?" or a similar question, reply with "I am Rishi's assistant built using a Large Language Model!"
# 3. **Handle unknowns:** If you cannot find the answer in the context, state "I don't know. Please ask Rishi on Discord at https://discord.gg/6ezpZGeCcM or email rishi@aiotsmartlabs.com." Do not make up an answer.
# 4. **Clarity and brevity:** Ensure your answers are clear and concise.

# CONTEXT: {context}

# QUESTION: {question}""")

from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# def warmup_model():
#     dummy_context = {"context": "dummy context", "question": "dummy question"}
#     rag_chain.invoke(dummy_context)

# # Call the warm-up function during startup
# warmup_model()

# Define the chat response function
def chatresponse(message, history):
    output = rag_chain.invoke(message)
    response = output.split('ANSWER: ')[-1].strip()
    return response

# css_code='body{background-image:url("https://picsum.photos/seed/picsum/200/300");}'
# css = ".gradio-container {background: url('file=https://i.imgur.com/u8isIYl.png')}"
# css = ".gradio-container {background: url('file=https://i.imgur.com/rwk7ykG.png')}"
# css = ".gradio-container {background: url('file=https://i.imgur.com/LAfi4yx.png')}"




# Launch the Gradio chat interface
gr.ChatInterface(
    chatresponse,
    textbox = gr.Textbox(placeholder="Type in your message"),
    title = "Chitti Chatbot: AIoT SMART Labs Assistant",
    description = "Ask Chitti any question about the organization, program, or projects. I'm using a free API with rate limits, so response may be slow sometimes",
    examples = ["What is the IoT Summer Program?", "I'm a 9th grader. Am I eligible for the program?", "What are the dates for the online & live batches?", 
                "What projects do we do in this summer program?", "What accounts do we need to create for the IoT Summer Program 2024"],
    theme = "base",
).launch()

# import gradio as gr
# from langchain_community.document_loaders import CSVLoader  # Changed import
# from langchain_community.vectorstores import FAISS  # Changed import
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.llms import HuggingFaceLLM  # Adjusted for correct instantiation
# import warnings
# from huggingface_hub import login
# import os
# from transformers import pipeline

# # Initialize the LLM using pipeline
# llm = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct")  # Adjusted initialization

# # Load CSV file
# loader = CSVLoader(file_path='aiotsmartlabs_faq.csv', source_column='prompt')
# data = loader.load()

# # Suppress warnings
# warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
# warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated")

# # Embedding model
# model_name = "BAAI/bge-m3"
# instructor_embeddings = HuggingFaceLLM(model_name=model_name)  # Adjusted for correct instantiation

# # Create FAISS vector store from documents
# vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
# retriever = vectordb.as_retriever()

# # Define the prompt template
# prompt_template = """Given the following context and a question, generate an answer based on the context only.
# In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
# If somebody asks "Who are you?" or a similar phrase, state "I am Rishi's assistant built using a Large Language Model!"
# If the answer is not found in the context, kindly state "I don't know. Please ask Rishi on Discord. Discord Invite Link: https://discord.gg/6ezpZGeCcM. Or email at rishi@aiotsmartlabs.com" Don't try to make up an answer.
# CONTEXT: {context}
# QUESTION: {question}"""

# PROMPT = PromptTemplate(
#     template=prompt_template, input_variables=["context", "question"]
# )

# # Initialize the RetrievalQA chain
# chain = RetrievalQA.from_chain_type(llm=llm,  # Adjusted initialization
#                                     chain_type="stuff",
#                                     retriever=retriever,
#                                     input_key="query",
#                                     return_source_documents=True,
#                                     chain_type_kwargs={"prompt": PROMPT})

# # Define the chat response function
# def chatresponse(message, history):
#     output = chain(message)
#     return output['result']

# # Launch the Gradio chat interface
# gr.ChatInterface(chatresponse).launch()


# import gradio as gr
# # from langchain.llms import GooglePalm
# from langchain_google_genai import GoogleGenerativeAI
# from langchain.document_loaders.csv_loader import CSVLoader
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# import warnings
# from huggingface_hub import login
# import os


# from transformers import pipeline
# llm = pipeline("feature-extraction", model="mixedbread-ai/mxbai-embed-large-v1") 

# # from transformers import AutoModel
# # llm = AutoModel.from_pretrained("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True) 

# # LLAMA
# # from transformers import AutoModelForCausalLM, AutoTokenizer
# # from transformers import pipeline

# # hf_token = os.environ['llama_token']

# # login(token=hf_token)

# # llm = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct")
# # llm = pipeline("text-generation", model = "meta-llama/Meta-Llama-3-70B-Instruct")

# # MISTRAL
# # llm = pipeline("text-generation", model="mistralai/Mixtral-8x22B-Instruct-v0.1")


# # TO USE GOOGLE MODELS
# # api_key = "AIzaSyCdM_aAIsW_nPbjarOF83mbX1_z1cVX2_M"

# # llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key)
# # llm = GooglePalm(google_api_key = api_key, temperature=0.7)

# # LOADING CSV FILE
# loader = CSVLoader(file_path='aiotsmartlabs_faq.csv', source_column = 'prompt')
# data = loader.load()

# # SUPPRESSING WARNINGS
# warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
# warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated")


# # EMBEDDING MODEL
# model_name = "BAAI/bge-m3"
# instructor_embeddings = HuggingFaceEmbeddings(model_name=model_name)

# # Create FAISS vector store from documents
# vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
# retriever = vectordb.as_retriever()

# prompt_template = """Given the following context and a question, generate an answer based on the context only.

# In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
# If somebody asks "Who are you?" or a similar phrase, state "I am Rishi's assistant built using a Large Language Model!"
# If the answer is not found in the context, kindly state "I don't know. Please ask Rishi on Discord. Discord Invite Link: https://discord.gg/6ezpZGeCcM. Or email at rishi@aiotsmartlabs.com" Don't try to make up an answer.

# CONTEXT: {context}

# QUESTION: {question}"""

# PROMPT = PromptTemplate(
#     template = prompt_template, input_variables = ["context", "question"]
# )


# chain = RetrievalQA.from_chain_type(llm = llm,
#             chain_type="stuff",
#             retriever=retriever,
#             input_key="query",
#             return_source_documents=True,
#             chain_type_kwargs = {"prompt": PROMPT})

# def chatresponse(message, history):
#     output = chain(message)
#     return output['result']

# gr.ChatInterface(chatresponse).launch()



# import gradio as gr

# # from langchain.llms import GooglePalm
# # from langchain.document_loaders.csv_loader import CSVLoader
# # from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain.vectorstores import FAISS


# from langchain_community.llms import GooglePalm
# from langchain_community.document_loaders import CSVLoader
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings


# api_key = "AIzaSyCdM_aAIsW_nPbjarOF83mbX1_z1cVX2_M"

# llm = GooglePalm(google_api_key = api_key, temperature=0.7)


# loader = CSVLoader(file_path='aiotsmartlabs_faq.csv', source_column = 'prompt')
# data = loader.load()


# instructor_embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-m3")
# vectordb = FAISS.from_documents(documents = data, embedding = instructor_embeddings)

# retriever = vectordb.as_retriever()

# from langchain.prompts import PromptTemplate

# prompt_template = """Given the following context and a question, generate an answer based on the context only.

# In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
# If somebody asks "Who are you?" or a similar phrase, state "I am Rishi's assistant built using a Large Language Model!"
# If the answer is not found in the context, kindly state "I don't know. Please ask Rishi on Discord. Discord Invite Link: https://discord.gg/6ezpZGeCcM. Or email at rishi@aiotsmartlabs.com" Don't try to make up an answer.

# CONTEXT: {context}

# QUESTION: {question}"""

# PROMPT = PromptTemplate(
#     template = prompt_template, input_variables = ["context", "question"]
# )

# from langchain.chains import RetrievalQA

# chain = RetrievalQA.from_chain_type(llm = llm,
#             chain_type="stuff",
#             retriever=retriever,
#             input_key="query",
#             return_source_documents=True,
#             chain_type_kwargs = {"prompt": PROMPT})

# def chatresponse(message, history):
#     output = chain(message)
#     return output['result']

# gr.ChatInterface(chatresponse).launch()


# import gradio as gr
# from langchain.llms import GooglePalm

# api_key = "AIzaSyCdM_aAIsW_nPbjarOF83mbX1_z1cVX2_M"

# llm = GooglePalm(google_api_key = api_key, temperature=0.7)

# from langchain.document_loaders.csv_loader import CSVLoader

# loader = CSVLoader(file_path='aiotsmartlabs_faq.csv', source_column = 'prompt')
# data = loader.load()

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS

# # instructor_embeddings = HuggingFaceEmbeddings(model_name = "Alibaba-NLP/gte-Qwen2-7B-instruct") # best model <-- but too big
# instructor_embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-m3")
# # instructor_embeddings = HuggingFaceEmbeddings()

# vectordb = FAISS.from_documents(documents = data, embedding = instructor_embeddings)

# # e = embeddings_model.embed_query("What is your refund policy")

# retriever = vectordb.as_retriever()

# from langchain.prompts import PromptTemplate

# prompt_template = """Given the following context and a question, generate an answer based on the context only.

# In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
# If somebody asks "Who are you?" or a similar phrase, state "I am Rishi's assistant built using a Large Language Model!"
# If the answer is not found in the context, kindly state "I don't know. Please ask Rishi on Discord. Discord Invite Link: https://discord.gg/6ezpZGeCcM. Or email at rishi@aiotsmartlabs.com" Don't try to make up an answer.

# CONTEXT: {context}

# QUESTION: {question}"""

# PROMPT = PromptTemplate(
#     template = prompt_template, input_variables = ["context", "question"]
# )

# from langchain.chains import RetrievalQA

# chain = RetrievalQA.from_chain_type(llm = llm,
#             chain_type="stuff",
#             retriever=retriever,
#             input_key="query",
#             return_source_documents=True,
#             chain_type_kwargs = {"prompt": PROMPT})

# # Load your LLM model and necessary components
# # Assume `chain` is a function defined in your notebook that takes a query and returns the output as shown
# # For this example, we'll assume the model and chain function are already available

# def chatbot(query):
#     response = chain(query)
#     # Extract the 'result' part of the response
#     result = response.get('result', 'Sorry, I could not find an answer.')
#     return result

# # Define the Gradio interface
# iface = gr.Interface(
#     fn=chatbot,  # Function to call
#     inputs=gr.inputs.Textbox(lines=2, placeholder="Enter your question here..."),  # Input type
#     outputs="text",  # Output type
#     title="Hugging Face LLM Chatbot",
#     description="Ask any question related to the documents and get an answer from the LLM model.",
# )

# # Launch the interface
# iface.launch()

# # Save this file as app.py and push it to your Hugging Face Space repository

# # import gradio as gr

# # def greet(name, intensity):
# #     return "Hello, " + name + "!" * int(intensity)

# # demo = gr.Interface(
# #     fn=greet,
# #     inputs=["text", "slider"],
# #     outputs=["text"],
# # )

# # demo.launch()


# # import gradio as gr
# # from huggingface_hub import InferenceClient

# # """
# # For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
# # """
# # client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


# # def respond(
# #     message,
# #     history: list[tuple[str, str]],
# #     system_message,
# #     max_tokens,
# #     temperature,
# #     top_p,
# # ):
# #     messages = [{"role": "system", "content": system_message}]

# #     for val in history:
# #         if val[0]:
# #             messages.append({"role": "user", "content": val[0]})
# #         if val[1]:
# #             messages.append({"role": "assistant", "content": val[1]})

# #     messages.append({"role": "user", "content": message})

# #     response = ""

# #     for message in client.chat_completion(
# #         messages,
# #         max_tokens=max_tokens,
# #         stream=True,
# #         temperature=temperature,
# #         top_p=top_p,
# #     ):
# #         token = message.choices[0].delta.content

# #         response += token
# #         yield response

# # """
# # For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
# # """
# # demo = gr.ChatInterface(
# #     respond,
# #     additional_inputs=[
# #         gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
# #         gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
# #         gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
# #         gr.Slider(
# #             minimum=0.1,
# #             maximum=1.0,
# #             value=0.95,
# #             step=0.05,
# #             label="Top-p (nucleus sampling)",
# #         ),
# #     ],
# # )


# # if __name__ == "__main__":
# #     demo.launch()
