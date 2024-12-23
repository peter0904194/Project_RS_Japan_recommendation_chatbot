import pandas as pd 
from langchain_community.document_loaders.csv_loader import CSVLoader
from pathlib import Path
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
import os
from dotenv import load_dotenv
import sys
from langchain.docstore.document import Document
from rank_bm25 import BM25Okapi
import numpy as np
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import pickle
import faiss
from fuzzywuzzy import fuzz
import re



# LangChain 관련 라이브러리
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# RAG 구현 관련 라이브러리
from langchain.vectorstores import FAISS  # 또는 다른 벡터 저장소
from langchain.embeddings import OpenAIEmbeddings  # OpenAI 임베딩
from langchain.llms import OpenAI  # OpenAI 모델 호출
import ast 


# 언급된 지명을 추출
def extract_locations(result_content, documents):
    # 지명 추출 (문서에 나올 가능성이 있는 이름을 기준으로)
    locations = [doc.page_content.split(":")[1].split(":")[0].strip() for doc in documents]
    
    # 사용된 문서와 지역 찾기
    matched_documents = []
    for location in locations:
        if location in result_content:
            for doc in documents:
                if fuzz.partial_ratio(location, doc.page_content) > 70:
                    region = doc.page_content.split("//")[1].replace("(위치:", "").replace(")", "").strip()
                    matched_documents.append({"name": location, "region": region})
                    break  # 하나의 문서만 매칭
    
    return matched_documents


def generate_query(location: str, food: str, user_preference: str) -> str:
    query_template = "위치: {location}, 음식: {food}, 조건: {preference}"
    query = query_template.format(location=location, food=food, preference=user_preference)
    return query

# 식당 특징 반환
def extract_features(data):
    extracted_features = []
    for item in data:
        # 문자열을 리스트로 변환
        parsed_list = ast.literal_eval(item)
        # 특징만 추출
        features = [feature[0] for feature in parsed_list]
        extracted_features.extend(features)
    return extracted_features

def extract_first_item(input_string):
        # 문자열을 리스트로 변환
        parsed_list = ast.literal_eval(input_string)
        
        # 첫 번째 항목의 문자열 추출
        if parsed_list and isinstance(parsed_list[0], tuple):
            return parsed_list[0][0].strip()


def sanitize_input(input_string):
    # 대괄호 개수 확인 및 보정
    open_brackets = input_string.count("[")
    close_brackets = input_string.count("]")
    if open_brackets > close_brackets:
        input_string += "]" * (open_brackets - close_brackets)
    return input_string
