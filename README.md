# Japan_recommenation_chatbot
2024년 2학기, 비타민 14기, 추천시스템 1조 


# 1. 프로젝트명
LLM을 활용한 일본 여행 추천 프로젝트 


# 2. 프로젝트 소개

저희 프로젝트는 **사용자 맞춤형 일본 여행 추천 챗봇**을 개발하는 데 중점을 둔 시스템입니다.

이 챗봇은 **사용자의 영화 시청 기록**과 **국내 음식점 리뷰 데이터**를 기반으로 개인의 성향과 선호도를 분석합니다.

1. **BERT4Rec 모델**을 사용하여 사용자가 선호하는 영화 장르와 분위기를 분석했습니다.
2. **AutoEncoder 모델**을 활용해 한국 음식점 리뷰 데이터를 통해 선호하는 음식 카테고리와 분위기를 파악했습니다.
3. 분석된 데이터를 **LLM**과 **RAG 알고리즘**을 통해 일본 여행지와 음식점 정보를 벡터 DB로 구성하여 개인화된 추천을 제공합니다.

### 주요 기능
- **여행지 추천**: 사용자가 좋아하는 영화의 분위기와 특징을 반영하여 일본 여행지를 추천합니다.
- **음식점 추천**: 한국 음식점 리뷰를 기반으로 일본 음식점 데이터를 매핑하여 사용자의 음식 취향에 맞는 추천을 제공합니다.
- **효율적인 응답**: 임베딩 기반 유사도 계산과 리랭킹 알고리즘을 통해 추천 속도를 개선하여 사용자 경험을 최적화했습니다.

### 기대 효과
- **개인화된 여행 경험**: 사용자가 명확히 인지하지 못한 취향까지 반영한 맞춤형 추천을 통해 여행 계획의 고민을 덜어줍니다.
- **최신 기술 활용**: BERT4Rec, AutoEncoder, LLM, RAG 등 최신 추천 시스템 기술을 통합하여 고도화된 추천 시스템을 구현했습니다.

저희 챗봇은 사용자의 성향과 니즈를 기반으로 **최적의 여행지와 음식점을 추천**함으로써 **여행 계획의 간편화와 선택의 어려움을 해결**하는 것을 목표로 합니다.


# 3. 알고리즘 흐름도
![image](https://github.com/user-attachments/assets/4b0b5af9-17b0-4453-9e94-50c1b53b7935)


# 4. 세부 알고리즘 소개
① BERT4Rec : 고객 성향 파악 - 사용자의 영화 취향에 숨어있는 성향을 분석해 여행지 추천에 반영하고자 함.

② Auto Encoder : 사용 데이터 선별 - 고객이 기존에 작성한 한국 음식점 리뷰 데이터를 바탕으로 일본 음식점 추천에 사용될 데이터를 선별함. 

![image](https://github.com/user-attachments/assets/1461a6b6-52c3-4887-b6c7-7864ee44a7c2)



③ LLM recommendation : ①, ②에서 추천 받은 영화 및 식당을 기반으로 RAG의 강점을 살려, 챗봇을 이용하여 정보를 제공할 수 있게 함. 

![image](https://github.com/user-attachments/assets/10ad0f81-d399-4fac-be1b-c1c67ed2622b)


![image](https://github.com/user-attachments/assets/d6f0ae2d-5881-4e76-b4f6-b62de89eec6b)



# 5. 시연 영상 
![KakaoTalk_20241223_231344296](https://github.com/user-attachments/assets/906e032d-9565-4329-a2c8-6de526c4b0bb)


---
## 논문 스터디 

## Contributors

| 순번  | 발표날짜      |     이름              | 모델명(노션 발표자료 링크)                                                                                                                                 | 논문명                                                                                                                                 |
|-----|-----------|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
|  1   | 2024.10.10  |김민열       |   [Bert4rec](https://www.notion.so/BERT4Rec-Sequential-Recommendation-with-Bidirectional-Encoder-Representations-from-Transformer-10eab9efd4d48035bd1dc673c7d175a4?pvs=4)       |     BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer    |
|  2   | 2024.10.10  |서준형       |      [VAE for Collaborative Filtering](https://www.notion.so/Variational-Autoencoders-for-Collaborative-Filtering-10eab9efd4d48078ad00df288d474c4a?pvs=4)    |     Variational Autoencoders for Collaborative Filtering  |
|  3   | 2024.10.10  |김선재       |   [Neural Collaborative Filtering](https://www.notion.so/Neural-Collaborative-Filtering-10eab9efd4d480018458fbee7e4947ed?pvs=4)             |  Neural Collaborative Filtering        |
|  4   | 2024.10.10  |김정현       | [Neural Graph Colaborative Filtering](https://www.notion.so/Neural-Graph-Collaborative-Filtering-10eab9efd4d480809c37ff0bef81af50?pvs=4)    | Neural Graph Collaborative Filtering   |
|  5   | 2024.10.10  |양태원       |    [Wide & Deep Learning for Recommender Sysytems](https://www.notion.so/Wide-Deep-Learning-for-Recommender-Systems-10eab9efd4d4802bb144c63620505b42?pvs=4)                 | Wide & Deep Learning for Recommender Systems  |                        
