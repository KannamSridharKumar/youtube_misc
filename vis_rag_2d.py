import os
import re
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

#os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

url = "https://www.paulgraham.com/worked.html"
response = requests.get(url)
essay_text = re.search(r'<font.*?>(.*?)</font>', response.text, re.DOTALL).group(1)
essay_text = re.sub(r'<.*?>', '', essay_text)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
chunks = text_splitter.split_text(essay_text)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(chunks, embeddings)

llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

all_embeddings = vectorstore.embedding_function.embed_documents(chunks)

pca = PCA(n_components=2)
all_embeddings_2d = pca.fit_transform(all_embeddings)

app = Dash(__name__)

app.layout = html.Div([
    html.H1("RAG Data Visualization", style={'textAlign': 'center'}),
    html.Div([
        dcc.Input(
            id="question-input",
            type="text",
            placeholder="Enter your question here",
            style={'width': '70%', 'height': '40px', 'fontSize': '16px', 'marginRight': '10px'}
        ),
        html.Button(
            'Submit',
            id='submit-button',
            n_clicks=0,
            style={'height': '40px', 'fontSize': '16px'}
        )
    ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),
    html.Div(id="answer-output", style={'marginBottom': '20px', 'textAlign': 'center', 'fontSize': '18px'}),
    dcc.Graph(id="embedding-plot")
])

@app.callback(
    [Output("answer-output", "children"),
     Output("embedding-plot", "figure")],
    [Input("submit-button", "n_clicks")],
    [State("question-input", "value")]
)
def update_output(n_clicks, question):
    if n_clicks > 0 and question:
        answer = qa_chain.run(question)
        
        docs = vectorstore.similarity_search(question, k=3)
        doc_texts = [doc.page_content for doc in docs]
        
        question_embedding = embeddings.embed_documents([question])[0]
        answer_embedding = embeddings.embed_documents([answer])[0]
        
        qa_embeddings_2d = pca.transform([question_embedding, answer_embedding])
        
        distances = euclidean_distances([question_embedding], all_embeddings)[0]
        
        max_distance = np.max(distances)
        normalized_sizes = 1 - (distances / max_distance)
        point_sizes = normalized_sizes * 18 + 7
        
        df = pd.DataFrame(all_embeddings_2d, columns=['x', 'y'])
        df['type'] = 'Corpus'
        df['size'] = point_sizes
        df['text'] = chunks
        
        qa_df = pd.DataFrame(qa_embeddings_2d, columns=['x', 'y'])
        qa_df['type'] = ['Question', 'Answer']
        qa_df['size'] = 10
        qa_df['text'] = [question, answer]
        df = pd.concat([df, qa_df], ignore_index=True)
        
        retrieved_indices = [chunks.index(doc) for doc in doc_texts]
        df.loc[retrieved_indices, 'type'] = 'Retrieved'

        df['text'] = df['text'].str.wrap(30)
        df['text'] = df['text'].apply(lambda x: x.replace('\n', '<br>'))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df[df['type'] == 'Corpus']['x'],
            y=df[df['type'] == 'Corpus']['y'],
            mode='markers',
            name='Corpus',
            marker=dict(size=df[df['type'] == 'Corpus']['size'], color='red', opacity=0.5),
            text=df[df['type'] == 'Corpus']['text'],
            hoverinfo='text'
        ))
        
        fig.add_trace(go.Scatter(
            x=df[df['type'] == 'Retrieved']['x'],
            y=df[df['type'] == 'Retrieved']['y'],
            mode='markers',
            name='Retrieved',
            marker=dict(size=df[df['type'] == 'Retrieved']['size'] * 1.5, color='green'),
            text=df[df['type'] == 'Retrieved']['text'],
            hoverinfo='text'
        ))
        
        for t in ['Question', 'Answer']:
            fig.add_trace(go.Scatter(
                x=df[df['type'] == t]['x'],
                y=df[df['type'] == t]['y'],
                mode='markers',
                name=t,
                marker=dict(size=24, symbol='diamond'),
                text=df[df['type'] == t]['text'],
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title="",
            height=800,
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
            hovermode='closest'
        )
        
        return html.Div([
            html.Strong("Question: "), html.Span(question),
            html.Br(),
            html.Strong("Answer: "), html.Span(answer)
        ]), fig
    
    return "Ask a question and click Submit", go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)