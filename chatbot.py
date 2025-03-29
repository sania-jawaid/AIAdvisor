import os
import time
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import numpy as np
from langchain.schema import Document
import faiss
import validators


# Use HuggingFaceEmbeddings for LangChain compatibility
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"trust_remote_code": True}
)

articles_db = []  # Store articles for retrieval

# Function to Clean and Validate URL
def clean_url(url):
    url = url.strip()  # Remove extra spaces
    if not validators.url(url):  # Check if URL is valid
        return None
    return url


# Function to create vector store
def create_vector_store1(texts):
    start_time = time.time()

    try:
        embeddings = embedding_model.embed_documents(texts)
    except Exception as e:
        print("Error while embedding documents:", str(e))
        raise ValueError("Embedding model failed to process documents.")

    docs = [Document(page_content=text) for text in texts if text.strip()]

    # Generate embeddings
    # embeddings = embedding_model.embed_documents([doc.page_content for doc in docs])
    # embeddings = [embedding_model.embed_query(text) for text in docs]

    # Generate embeddings for the document content
    # texts = [doc.page_content for doc in docs]
    # embeddings = embedding_model.encode(texts)

    # Generate embeddings using LangChain's HuggingFaceEmbeddings
    # embeddings = embedding_model.embed_documents(texts)

    # Convert embeddings to NumPy array for FAISS
    embeddings = np.array(embeddings, dtype=np.float32)

    if embeddings.size == 0:
        print("Embedding array is empty. Check if documents contain valid text.")

    if embeddings.ndim == 1:  # If embeddings are 1D, reshape to 2D
        embeddings = embeddings.reshape(-1, len(embeddings))


    # Get the embedding dimension dynamically
    dimension = embeddings.shape[1]  # Number of features in each vector

    # Create a FAISS index
    index = faiss.IndexFlatL2(dimension)
    # index.add(embeddings)
    index.add(np.array(embeddings))
    # articles_db.extend(texts)

    vectorstore = FAISS.from_documents(docs, embedding_model)
    # Wrap in LangChain FAISS object
    # vectorstore = FAISS(embedding_model, index)

    # Streamlit progress updates
    st.sidebar.write("Vector DB is ready")
    end_time = time.time()
    st.sidebar.write(f"Time taken to create DB: {end_time - start_time:.2f} seconds")

    return vectorstore, index

# Function to load and process the documents from a URL
def get_docs_from_url(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = text_splitter.split_documents(docs)
    st.sidebar.write('Documents Loaded from URL')
    return split_docs

def extract_insights_from_excel(df):

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort data by date
    df = df.sort_values(by='Date')

    # Display data summary
    print(df.head())
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(30).std()
    df['Momentum'] = df['Close'] - df['Close'].shift(10)  # 10-day momentum
    return df

def get_excel(uploaded_file):
    start_time = time.time()
    with open("temp.xlsx", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # loader = PyPDFLoader("temp.pdf")
    # documents = loader.load()s
    # Load the Excel file
    df = pd.read_csv(uploaded_file)
    df = extract_insights_from_excel(df)
    # Convert Nifty 50 data into text for embeddings
    # df['text'] = df.apply(lambda row: f"Date: {row.Date}, Close: {row.Close}, P/E: {row['P/E']}, P/B: {row['P/B']}, Div Yield: {row['Div Yield %']}", axis=1)

    df['text'] = df.apply(lambda row: f"Date: {row.Date}, Close: {row.Close}, P/E: {row['P/E']}, P/B: {row['P/B']}, Div Yield: {row['Div Yield %']}, SMA_50: {row['SMA_50']}, SMA_200: {row['SMA_200']}, Volatility: {row['Volatility']}, Momentum: {row['Momentum']}", axis=1)
    # Create embeddings
    # embeddings = embedding_model.encode(df['text'].tolist())
    vectorstore = FAISS.from_documents(df, embedding_model)
    # Store embeddings in FAISS
    # dimension = embeddings.shape[1]
    # index = faiss.IndexFlatL2(dimension)
    # index.add(np.array(embeddings))

    # Save index
    # faiss.write_index(index, "nifty50_faiss.index")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    final_documents = text_splitter.split_documents(documents)
    st.sidebar.write('Documents Loaded')
    end_time = time.time()
    st.sidebar.write(f"Time taken to load documents: {end_time - start_time:.2f} seconds")
    os.remove("temp.pdf")  # Clean up the temporary file
    return final_documents

# Function to load and process the documents from an uploaded PDF file
def get_docs(uploaded_file):
    start_time = time.time()
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    final_documents = text_splitter.split_documents(documents)
    st.sidebar.write('Documents Loaded')
    end_time = time.time()
    st.sidebar.write(f"Time taken to load documents: {end_time - start_time:.2f} seconds")
    os.remove("temp.pdf")  # Clean up the temporary file
    return final_documents

# # Function to create vector store
def create_vector_store(docs):
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"trust_remote_code": True})
    vectorstore = FAISS.from_documents(docs, embeddings)
    st.sidebar.write('DB is ready')
    end_time = time.time()
    st.sidebar.write(f"Time taken to create DB: {end_time - start_time:.2f} seconds")
    return vectorstore

# Function to interact with Groq AI
def chat_groq(messages):
    load_dotenv()
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    response_content = ''
    stream = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        max_tokens=1024,
        temperature=1.3,
        stream=True,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            response_content += chunk.choices[0].delta.content
    return response_content

# Function to summarize the chat history
def summarize_chat_history(chat_history):
    chat_history_text = " ".join([f"{chat['role']}: {chat['content']}" for chat in chat_history])
    prompt = f"Summarize the following chat history:\n\n{chat_history_text}"
    messages = [{'role': 'system', 'content': 'You are very good at summarizing the chat between User and Assistant'}]
    messages.append({'role': 'user', 'content': prompt})
    summary = chat_groq(messages)
    return summary

# def add_articles_to_faiss(articles):
#     texts = [article['content'] for article in articles]
#     embeddings = embedding_model.encode(texts)
#     index.add(np.array(embeddings))
#     return articles

# Scrape article links
def get_latest_articles(url):
    url = "https://www.moneycontrol.com/news/business/markets/wall-street-indices-plunge-into-losses-on-selling-in-tech-stocks-tariff-worries-12976944.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract links (modify based on structure)
    links = []
    for link in soup.find_all('a', href=True):
        if "/article/" in link['href'] and link['href'].startswith("http"):
            links.append(link['href'])

    print("links")
    print(links)
    return list(set(links[:5]))  # Return top 5 articles

# Function to retrieve relevant articles
def retrieve_relevant_articles(query, index, top_k=3):
    query_embedding = embedding_model.embed_documents([query])
    D, I = index.search(np.array(query_embedding), top_k)
    results = [articles_db[i] for i in I[0] if i < len(articles_db)]
    return results

# Function to fetch and store articles in FAISS
def add_articles_to_faiss(url):
    global articles_db
    article_links = get_latest_articles(url)
    
    articles = []
    for link in article_links:
        try:
            article = Article(link)
            article.download()
            article.parse()
            articles.append({"title": article.title, "content": article.text, "url": link})
            print("article")
            print(article)
        except:
            continue  # Skip failed articles

    # Store in FAISS
    texts = [article['content'] for article in articles]
    # embeddings = embedding_model.encode(texts)
    # embeddings = embedding_model.embed_documents(texts)
    vector_store, index = create_vector_store(texts)
  
    return articles, index

def rag_pipeline(prompt, max_length=200, do_sample=False):
    messages = [{'role': 'system', 'content': 'You are a very helpful assistant'}]
    messages.append({'role': 'user', 'content': prompt})
    try:
        ai_response = chat_groq(messages)
    except Exception as e:
        st.error(f"Error occurred during chat_groq execution: {str(e)}")
        ai_response = "An error occurred while fetching response. Please try again."
    return ai_response

# Function to summarize articles
def summarize_article(article_title, index):
    retrieved_articles = retrieve_relevant_articles(article_title, index)
    context = "\n".join([article["content"] for article in retrieved_articles])

    prompt = f"""
    Summarize the following financial news articles:

    {context}

    Provide a clear and concise summary.
    """
    
    summary = rag_pipeline(prompt, max_length=200, do_sample=False)[0]['generated_text']
    return summary

# Main function to control the app
def main():
    st.set_page_config(page_title='AIAdvisor')

    if "show_section" not in st.session_state:
        st.session_state.show_section = False

    # if st.button("Toggle Section"):
        # st.session_state.show_section = not st.session_state.show_section

    st.title("Personal AI Advisor")


    if st.session_state.show_section:
        st.subheader("This section is visible!")
        st.write("Click the button to toggle visibility.")

        st.title("📊 Indian Financial News Summarizer")

        # Define news sources
        NEWS_SOURCES = {
            "Economic Times": "https://economictimes.indiatimes.com/markets",
            "Moneycontrol": "https://www.moneycontrol.com/news/",
            "Business Standard": "https://www.business-standard.com/latest-news"
        }

        # Example: Adding articles
        articles_db = [
            {"title": "Nifty Rises Amid Market Optimism", "content": "The Nifty 50 rose by 1.5% as investors showed optimism..."},
            {"title": "RBI Updates on Inflation", "content": "RBI announced a 0.25% rate hike to curb inflationary pressures..."}
        ]
        # Store in FAISS
        # vector_store, index = create_vector_store(articles)
        texts = [article['content'] for article in articles_db]  # Extract only text
        # vector_store, index = create_vector_store(texts)

        # Select source
        source = st.selectbox("Choose a news source", list(NEWS_SOURCES.keys()), disabled=True)

        # Fetch and display articles
        if st.button("🔍 Fetch News", disabled=True):
            articles, index = add_articles_to_faiss(NEWS_SOURCES[source])
            # summary = summarize_article(article_url, index)
            if articles:
                st.write("### ✅ Articles Stored for Retrieval:")
                for i, article in enumerate(articles):
                    st.write(f"{i+1}. [{article['title']}]({article['url']})")
            else:
                st.warning("No articles found!")

        # Summarize selected article
        article_url = st.text_input("Paste article URL to summarize")

        if st.button("📜 Summarize", disabled=True):
        # query = st.text_input("🔍 Enter Article Title or Topic")
            if article_url:
                summary = summarize_article(article_url, index)
                st.write("### 📰 Summary:")
                st.write(summary)
            else:
                st.warning("Please enter a valid topic or title!")

    with st.expander("Instructions to upload Text PDF/URL"):
        st.write("1. Pull up the side bar in top left corner.")
        st.write("2. If uploading a PDF, click 'Upload PDF', select your file, and wait for 'Documents Loaded' confirmation.")
        st.write("3. If entering a web URL, enter the URL, click 'Enter Web URL', and submit 'Process URL' and wait for 'Documents Loaded from URL' confirmation.")
        st.write("4. After loading documents, click 'Create Vector Store' to process.Documents can only be uploaded once per session")
        st.write("5. Enter a question in the text area and submit to interact with the AI chatbot.")
        st.write("6. Click on Generate Chat Summary to get the conversation of the Chat Session.")

    # Sidebar for document source selection
    st.sidebar.subheader("Choose document source:")
    option = st.sidebar.radio("Select one:", ("Upload PDF", "Enter Web URL"))

    if "docs" not in st.session_state:
        st.session_state.docs = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt = ""
    if "chat_summary" not in st.session_state:
        st.session_state.chat_summary = ""

    if option == "Upload PDF":
        uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
        if uploaded_file is not None:
            if st.session_state.docs is None:
                with st.spinner("Loading documents..."):
                    docs = get_docs(uploaded_file)
                st.session_state.docs = docs

    if option == "Upload EXCEL":
        uploaded_file = st.sidebar.file_uploader("Upload an EXCEL file", type=["csv","xlsx", "xls"])
        if uploaded_file is not None:
            if st.session_state.docs is None:
                with st.spinner("Loading documents..."):
                    # docs = get_excel(uploaded_file)
                    st.error("Work in Progress! 🚨")
                # st.session_state.docs = docs

    elif option == "Enter Web URL":
        url = st.sidebar.text_input("Enter URL", key="url_input")
        if st.session_state.url_input != url:
            st.session_state.url_input = url
            st.session_state.docs = None
        if st.sidebar.button('Process URL'):
            if url and st.session_state.docs is None:
                with st.spinner("Fetching and processing documents from URL..."):
                    docs = get_docs_from_url(url)
                st.session_state.docs = docs

    if st.session_state.docs is not None:
        if st.sidebar.button('Create Vector Store'):
            with st.spinner("Creating vector store..."):
                vectorstore = create_vector_store(st.session_state.docs)
            st.session_state.vectorstore = vectorstore

    # if st.session_state.vectorstore is not None:
    if True:
        def submit_with_doc():
            user_message = st.session_state.user_input
            if user_message:
                retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                context = retriever.invoke(user_message)
                prompt = f'''
                Answer the user's question based on the latest input provided in the chat history. Ignore
                previous inputs unless they are directly related to the latest question. Provide a generic
                answer if the answer to the user's question is not present in the context by mentioning it
                as general information.

                Context: {context}

                Chat History: {st.session_state.chat_history}

                Latest Question: {user_message}
                '''

                messages = [{'role': 'system', 'content': 'You are a very helpful assistant'}]
                messages.append({'role': 'user', 'content': prompt})

                try:
                    ai_response = chat_groq(messages)
                except Exception as e:
                    st.error(f"Error occurred during chat_groq execution: {str(e)}")
                    ai_response = "An error occurred while fetching response. Please try again."

                # Display the current output prompt
                st.session_state.current_prompt = ai_response

                # Update chat history
                st.session_state.chat_history.append({'role': 'user', 'content': user_message})
                st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})

                # Clear the input field
                st.session_state.user_input = ""

    def submit_without_doc():
        user_message = st.session_state.user_input
        if user_message:
            prompt = f'''
            Answer the user's question based on the latest input provided in the chat history. Ignore
            previous inputs unless they are directly related to the latest
            question. 
            
            Chat History: {st.session_state.chat_history}

            Latest Question: {user_message}
            '''

            messages = [{'role': 'system', 'content': 'You are a very helpful assistant'}]
            messages.append({'role': 'user', 'content': prompt})

            try:
                ai_response = chat_groq(messages)
            except Exception as e:
                st.error(f"Error occurred during chat_groq execution: {str(e)}")
                ai_response = "An error occurred while fetching response. Please try again."

            # Display the current output prompt
            st.session_state.current_prompt = ai_response

            # Update chat history
            st.session_state.chat_history.append({'role': 'user', 'content': user_message})
            st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})

            # Clear the input field
            st.session_state.user_input = ""

    st.text_area("Enter your question:", key="user_input")
    if st.session_state.vectorstore is not None:
        st.button('Submit', on_click=submit_with_doc)  
    else:
        st.button('Submit', on_click=submit_without_doc)

    # Display the current output prompt if available
    if st.session_state.current_prompt:
        st.write(st.session_state.current_prompt)

    # Button to generate chat summary
    if st.button('Generate Chat Summary'):
        st.session_state.chat_summary = summarize_chat_history(st.session_state.chat_history)

    # Display the chat summary if available
    if st.session_state.chat_summary:
        with st.expander("Chat Summary"):
            st.write(st.session_state.chat_summary)

    # Display the last 4 messages in an expander
    with st.expander("Recent Chat History"):
        recent_history = st.session_state.chat_history[-8:][::-1]
        reversed_history = []
        for i in range(0, len(recent_history), 2):
            if i+1 < len(recent_history):
                reversed_history.extend([recent_history[i+1], recent_history[i]])
            else:
                reversed_history.append(recent_history[i])
        for chat in reversed_history:
            st.write(f"{chat['role'].capitalize()}: {chat['content']}")

if __name__ == "__main__":
    main()