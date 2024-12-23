import os
import sys
from dotenv import load_dotenv
from langchain.docstore.document import Document
from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate  # 추가






sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..'))) # Add the parent directory to the path sicnce we work with notebooks
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
class RatingScore(BaseModel):
    relevance_score: float = Field(..., description="The relevance score of a document to a user's movie preference.")

def rerank_documents_with_similarity(
    movie_title: str,
    movie_genres: str,
    movie_plot: str,
    docs: List[Document],
    top_n: int = 3
) -> List[Tuple[Document, float]]:
    """
    Filter documents based on similarity to the movie's genre and plot, then rerank using LLM.
    """
    # Step 1: Create query from movie information
    query = f"{movie_title}. Genre: {movie_genres}. Plot: {movie_plot}"

    # Step 2: Initialize a Korean SentenceTransformer model
    model = SentenceTransformer('./local_model')
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode([doc.page_content for doc in docs], convert_to_tensor=True)

    # Step 3: Calculate cosine similarities
    cosine_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings).squeeze()

    # Step 4: Sort documents by similarity
    scored_docs = [(doc, score.item()) for doc, score in zip(docs, cosine_scores)]
    scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:10]  # Take top 10 for reranking

    # Step 5: Rerank with LLM
    prompt_template = PromptTemplate(
        input_variables=["movie_genres", "movie_plot", "doc"],
        template="""On a scale of 1-10, rate the relevance of the following document to the user's inferred travel preference from their favorite movie. 
        Consider the overall mood, setting, and activities the user might enjoy based on the movie, rather than direct keyword matches.

        User's movie preference:
        - Movie Genres: {movie_genres}
        - Movie Plot: {movie_plot}

        Document (Travel Destination): {doc}

        Relevance Score:
        """
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)
    llm_chain = prompt_template | llm.with_structured_output(RatingScore)

    def evaluate_doc(doc: Document, score: float) -> Tuple[Document, float]:
        """Evaluate the relevance of a single document using the LLM."""
        input_data = {
            "movie_genres": movie_genres,
            "movie_plot": movie_plot,
            "doc": doc.page_content
        }
        try:
            result = llm_chain.invoke(input_data)
            llm_score = float(result.relevance_score)
        except (ValueError, AttributeError):
            llm_score = 0  # Default score if parsing fails
        # Combine initial similarity score with LLM score for final ranking
        combined_score = (llm_score * 0.7) + (score * 0.3)
        return (doc, combined_score, score)

    # Step 6: Apply LLM reranking
    reranked_docs = [evaluate_doc(doc, score) for doc, score in scored_docs]

    # Step 7: Sort by final combined score
    reranked_docs = sorted(reranked_docs, key=lambda x: x[1], reverse=True)

    return reranked_docs[:top_n]