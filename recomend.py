from rerank import rerank_documents_with_similarity
from rag_final import extract_features, generate_query, extract_locations, extract_first_item
from dotenv import load_dotenv
import os
import pandas as pd
import faiss
import pickle
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# .env 파일 불러오기
load_dotenv()

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되어 있지 않습니다.")


def recommend_travel(user_id, travel_preference):
    # 데이터 로드
    movie_faiss = "faiss_index_movie.index"
    embeddings = OpenAIEmbeddings()
    df1 = pd.read_csv("rag_data/user_movie.csv")
    loaded_index = faiss.read_index(movie_faiss)
    
    with open("index_to_docstore_id.pkl", "rb") as f:
        index_to_docstore_id = pickle.load(f)
    with open("docstore.pkl", "rb") as f:
        docstore = pickle.load(f)
    
    vector_store = FAISS(
        embedding_function=embeddings,
        index=loaded_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
    # 사용자 영화 정보 가져오기
    user_id = str(user_id)
    movie = df1[df1['user_id'] == user_id]
    movie_title = movie['title'].values[0]
    movie_genres = movie['genres'].values[0]
    movie_plot = movie['plot'].values[0]
    
    # 문서 검색
    document = retriever.invoke(travel_preference)
    
    reranked_docs = rerank_documents_with_similarity(
        movie_title=movie_title,
        movie_genres=movie_genres,
        movie_plot=movie_plot,
        docs=document,
        top_n=5
    )
    context = "\n".join([doc.page_content for doc, _, _ in reranked_docs])
    
    # 프롬프트 작성 및 RAG 실행
    system_prompt = """
당신은 일본 여행 추천 도우미입니다. 사용자가 좋아하는 영화와 여행 취향을 바탕으로 적합한 여행지를 추천하세요.  
추천 이유는 영화와 여행지의 분위기, 테마, 또는 사용자 취향과의 연결성을 중심으로 간결하고 설득력 있게 설명해주세요.

## 응답 구조 ##
1. 사용자의 영화 관심사와 여행 취향을 간단히 요약하며 대화를 시작하세요.
2. 추천 여행지 하나를 구체적으로 제시하고, 추천 이유를 자세하게 설명하세요.
3. 영화와의 연결성이 명확하지 않으면, 사용자의 여행 취향에 초점을 맞춰 추천 이유를 작성하세요.

사용자의 영화 정보:
- 제목: {movie_title}
- 장르: {movie_genres}
- 줄거리: {movie_plot}

사용자의 여행 취향:
{query}

추천 가능한 여행지 정보:
{context}
"""
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
    rag_chain = prompt | llm
    
    # RAG 실행 및 결과 추출
    result = rag_chain.invoke({
        "movie_title": movie_title,
        "movie_genres": movie_genres,
        "movie_plot": movie_plot,
        "query": travel_preference,
        "context": context
    })

    # RAG 결과 내용 출력
    print(result.content)

    # 여행지 정보 추출
    location = extract_locations(result.content, document)

    # 결과에서 중복 제거 및 필요한 정보 추출
    # `result.content`에서 첫 번째 소개 문장과 나머지 상세 설명을 분리
    result_lines = result.content.split("\n\n")
    movie_intro = result_lines[0] if len(result_lines) > 0 else "영화 소개를 가져올 수 없습니다."
    travel_details = "\n\n".join(result_lines[1:]) if len(result_lines) > 1 else "여행지 설명을 가져올 수 없습니다."

    # 반환할 데이터 구성
    travel_data = {
        "movie_title": movie_title,
        "movie_description": movie_intro,  # 영화 소개 부분만 저장
        "travel_name": location[0]["name"] if location else "추천된 여행지 없음",
        "travel_address": location[0]["region"] if location else "주소 정보 없음",
        "travel_explanation": travel_details  # 여행지 상세 설명만 저장
    }
    return travel_data
    # return travel_data["travel_address"]
    

def recommend_restaurant(user_id, restaurant_preference, final_location):
    # 데이터 로드
    
    kr_rest = pd.read_csv("rag_data/kr_df.csv")
    jp_rest = pd.read_csv("rag_data/jp_rest.csv")
    
    persist_directory = "japan_chroma_db_with_header"
    vector_store_japans = Chroma(
        collection_name="rag_japan",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=persist_directory
    )
    retriever = vector_store_japans.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    
    # 사용자 식당 정보 가져오기
    user_id = str(user_id)
    kr_rest_id = kr_rest.query('user == @user_id')
    if kr_rest_id.empty:
        return "사용자 ID에 대한 데이터가 없습니다."
    
    if len(kr_rest_id['recommended_item'].values) == 0:
        return "추천할 식당 정보가 없습니다."
    
    restuarant_style = kr_rest_id['recommended_item'].values[0]
    restuarant_info = kr_rest_id['more_info_top'].values[0]
    restuarant_people = kr_rest_id['상황/인물_top'].values[0]
    
    info = extract_first_item(restuarant_info)
    people = extract_first_item(restuarant_people)
    food = restuarant_style

    if restaurant_preference:  # 사용자가 입력한 경우
        if ',' in restaurant_preference:  # ','로 구분된 음식과 식당 특징이 모두 입력된 경우
            food, user_preference = map(str.strip, restaurant_preference.split(',', 1))
        else:  # 음식 종류 없이 식당 특징만 입력된 경우
            food = restuarant_style  # 기본값
            user_preference = restaurant_preference.strip()  # 입력된 내용을 식당 특징으로 사용
    else:  # 사용자가 아무것도 입력하지 않은 경우
        food = restuarant_style  # 기본값
        user_preference = "맛있는 식당"  # 필요 시 기본값 설정


    # 쿼리 생성 및 검색
    query = generate_query(final_location, food, user_preference)
    res_list = retriever.invoke(query)
    if not res_list:
        return "검색된 식당이 없습니다."

    # 프롬프트 작성 및 RAG 실행
    system_prompt2 = """
당신은 일본 식당 추천 도우미입니다. 사용자가 한국에서 선호하던 식당의 특징을 간단히 언급하며 대화를 시작하세요. (ex 한국에서 {food}와 같은 요리를 좋아하셨군요! )
사용자의 취향을 바탕으로 일본에서 찾을 수 있는 식당을 추천하고, 추천 이유와 함께 설명을 덧붙이세요. 
가능하면, 사용자의 취향과 잘 맞는 옵션을 선택해 더 나은 식사 경험을 도와주세요. 

특히, 아래의 조건을 반드시 준수하여 추천하세요:
1. 쿼리에 명시된 **위치**에 있는 식당만 추천합니다.
2. 쿼리에 명시된 **음식**을 우선적으로 추천합니다.
3. 만약 해당 음식이 없다면, 비슷한 종류의 음식을 추천하고 그 이유를 설명합니다.

## 사용자의 식당 취향 정보 ##
- 함께 가던 사람: {people}
- 식당 정보: {info}

## 사용자가 원하는 식당 조건 ##
{query}

아래는 사용자의 취향을 바탕으로 검색된 일본 식당 정보입니다. 
이 정보를 바탕으로 사용자가 만족할 만한 식당을 추천하고, 다음 사항을 포함하여 친절히 설명하세요:
1. 추천 식당 이름
2. 추천 이유 (특히 쿼리와의 적합성 설명)
3. 해당 음식이 없는 경우, 비슷한 음식 추천과 그 이유

{res_list}

"""

    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt2)])
    rag_chain = prompt | llm
    
    # RAG 실행 및 결과 추출
    try:
        result = rag_chain.invoke({
            "people": people,
            "info": info,
            "food": food,
            "query": query,
            "res_list": res_list
        })

        # 결과를 문자열로 반환
        if result and hasattr(result, "content"):
            return {"restaurant_explanation": result.content}  # JSON 형식으로 반환
        else:
            return {"error": "추천 결과를 가져올 수 없습니다."}

    except Exception as e:
        return {"error": f"식당 추천 중 오류 발생: {str(e)}"}