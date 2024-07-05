# A Brief About Chitti Chatbot: AIoT SMART Labs Assistant
Chitti is a chatbot that can help you answer questions about the IoT Summer Program, Projects in the program curriculum, Innovations of AIoT SMART Labs, Certification process and more! Ask Chitti at [Chitti Chatbot: AIoT SMART Labs Assistant](https://www.aiotsmartlabs.com/innovations/chitti-chatbot)

Chitti Chatbot is a retrieval-augmented-generation (RAG) application which utilizes a Mistral Large Language Model (LLM) for generation and a bge-m3 model developed by BAAI for retrieval. The application is built using:
- [LangChain](https://www.langchain.com/langchain), an open-source framework designed to simplify the creation of applications using LLMs
- [Gradio](https://www.gradio.app/), an open-source python package to build a user interface for the chatbot. I started with Streamlit, but I later shifted to Gradio as it seems to be more popular nowadays.
- [HuggingFace](https://huggingface.co/), a community and platform focused on field of AI and Natural Language Processing (NLP). Used to access open-source LLM models and host the chatbot.


# Chitti Architecture
![image](https://github.com/rishisim/Chitti-Chatbot/assets/86998121/42adb6a6-9523-45ee-b2f2-ff747f8a8e28)

## What is RAG?
RAG is a technique for augmenting an LLM's knowledge with additional data. LLMs can reason about common, well-known topics, but the knowledge is limited to the public data that they are trained on. The process of building an application that can reason about specific data outside of the LLM's current knowledge is Retrieval Augmented Generation (RAG). Chitti Chatbot is a RAG application.

**3 Components of Chitti Chatbot: Indexing, Retrieval, Generation**

### How I Converted AIoT SMART Labs Website Into a Database (Indexing)
I used the Frequently Asked Questions (FAQ) page as data for the model. Additionally, I converted previous Summer Program Videos and the Brochure to text and converted them into a Question/Answer format to append to the database. This Question/Answer formatted database is converted into vector embeddings for the model to understand and search through.

![image](https://github.com/rishisim/Chitti-Chatbot/assets/86998121/8ab3ead6-3ee8-415d-ac82-2e311938a4d3)

### Retrieving Relevant Information (Retrieval)
Based on the query (question of the user), the retrieval model fetches the relevant information from the database and inputs the curated Q/A set into the Prompt Template. After retrieval, the Prompt Template is filled with a string of text that combines my custom Instructions, the context (curated Q/A set), and the user query.

![image](https://github.com/rishisim/Chitti-Chatbot/assets/86998121/b16f3175-a4cb-40cb-9668-563106264df8)

### Generating A Response (Generation)
The completed Prompt is passed to the Large Language Model (LLM) which generates a coherent and contextually appropriate response using the retrieved information.

![image](https://github.com/rishisim/Chitti-Chatbot/assets/86998121/7fb315d4-aa0b-4217-8aa6-854755e58870)

# Decisions During the Process of Making Chitti
## Choice of Model
For Chitti, as a RAG application, it requires a Retrieval Model and a Generation Model. The retrieval model must excel at fetching relevant information from a large database and the generation model should create appropriate and detailed responses. HuggingFace provides leaderboards with information on performance of models at various benchmarks.

**MTEB Leaderboard (Retrieval Model)** 

[The Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) shows models evaluated at different tasks related to embedding. The leaderboard shows models evaluated at various tasks such as Classification, Summarization, Retrieval and more. For Retrieval, I had several text embedding models to choose from (e.g. models from Mistral AI, Meta's Llama series, etc).

**Open LLM Leaderboard (Generation Model)**

[The Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) shows open-source models evaluated on 6 key benchmarks using Eleuther AI Language Model Evaluation Harness, a framework to test generative models on various tasks. In the leaderboard, the Large Language Models can range from having a few hundred million parameters to 140 billion parameters.

**Local v. API**

Local v. API is a dillema that I had when choosing the model. If I choose a local model, the model's capability is limited by its size (the HuggingFace space has only 50GB of space). However, if I choose an API approach, the HuggingFace Serverless Inference API has to support the model and there are rate limits for the free version of the API. 

**Final Step: Choosing the Model**

With all the above information in mind, I tested various different models. For text-embedding models, I evaluated the model's ability to retrieve Q/A pairs relevant to the query. I realized that there was no significant performance and capability difference between embedding models of different size. Therefore, smaller models can perform as well as large models.

For Large Language Models, I tested Meta's LLama models, Google's Gemma, and Mistral AI models. I realized that there are significant performance differences between Base Models and Instruct fine-tuned Models for all of the above models. Models also have significant performance differences based on the number of parameters. To optimize the response quality, I choose Instruct Models with relatively high parameter size. Due to the high parameter size, I opted for an API approach for the generative models instead of downloading the model locally. When comparing the Instruct models for LLama, Gemma and Mistral, I found that the Mistral-7B-Instruct-v0.3 Large Language Model suits my requirements for generating concise and detailed responses. Other models would often end in incomplete sentences or vaguely answer the question.

**Model Parameters**

Regarding the model parameters, I set
- The temperature = 0.7. Giving the model some creativity while maintaining the topic of the question in its response.
- The Top_P = 0.9. Ensures contextually relevant and coherent text.
- The max_new_tokens = 256. Response length will stay within this number of tokens.

Note that all above information are personal experiences with my personal use case that I am testing. The observations may not be generalized to every situation.

## Future Work
To improve the Chitti Chatbot, I can add conversation history to the chatbot context so users can maintain a conversation about a certain topic over several messages.
