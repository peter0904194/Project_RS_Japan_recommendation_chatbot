# Japan_recommenation_chatbot
2024년 2학기, 비타민 14기, 추천시스템 1조 

# 1. 프로젝트명
LLM을 활용한 일본 여행 추천 프로젝트 

# 2. 프로젝트 소개
꾸준한 일본 여행 수요 증가와 2030 세대의 챗봇 사용량 증가를 배경으로, 일본 여행 장소와 음식점을 추천하는 서비스를 구현했습니다. 다양한 추천 시스템 기법을 적용하고, 챗봇을 구현하여 추천 서비스를 쉽게 이용할 수 있도록 하였습니다. 

# 3. 알고리즘 흐름도
![image](https://github.com/user-attachments/assets/4b0b5af9-17b0-4453-9e94-50c1b53b7935)

# 4. 세부 알고리즘 소개
① BERT4Rec : 고객 성향 파악 - 사용자의 영화 취향에 숨어있는 성향을 분석해 여행지 추천해 반영하고자 함.

② Auto Encoder : 사용 데이터 선별 - 고객이 기존에 작성한 한국 음식점 리뷰 데이터를 바탕으로 일본 음식점 추천에 사용될 데이터를 선별함. 
![image](https://github.com/user-attachments/assets/5db5243b-8a4c-46b5-9cd4-d2969e861139)

③ LLM recommendation : ①, ②에서 추천 받은 영화 및 식당을 기반으로 RAG의 강점을 살려, 챗봇을 이용하여 정보를 제공할 수 있게 함. 
<img width="527" alt="image" src="https://github.com/user-attachments/assets/1cb1f41c-0d33-4fd3-b6eb-f1637826cb8d" />



# 5. 시연 영상 

---
## 논문 스터디 

## Contributors

| 순번  | 발표날짜      |     이름           | 모델명(노션 발표자료 링크)                                                                                                                                    | 논문명                                                                                                                                 |
|-----|-----------|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
|  1   | 2024.10.10  |김민열       |   [Bert4rec](https://www.notion.so/BERT4Rec-Sequential-Recommendation-with-Bidirectional-Encoder-Representations-from-Transformer-10eab9efd4d48035bd1dc673c7d175a4?pvs=4)       |     BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer    |
|  2   | 2024.10.10  |서준형       |      [VAE for Collaborative Filtering](https://www.notion.so/Variational-Autoencoders-for-Collaborative-Filtering-10eab9efd4d48078ad00df288d474c4a?pvs=4)    |     Variational Autoencoders for Collaborative Filtering  |
|  3   | 2024.10.10  |김선재       |   [Neural Collaborative Filtering](https://www.notion.so/Neural-Collaborative-Filtering-10eab9efd4d480018458fbee7e4947ed?pvs=4)             |  Neural Collaborative Filtering        |
|  4   | 2024.10.10  |김정현       | [Neural Graph Colaborative Filtering](https://www.notion.so/Neural-Graph-Collaborative-Filtering-10eab9efd4d480809c37ff0bef81af50?pvs=4)    | Neural Graph Collaborative Filtering   |
|  5   | 2024.10.10  |양태원       |    [Wide & Deep Learning for Recommender Sysytems](https://www.notion.so/Wide-Deep-Learning-for-Recommender-Systems-10eab9efd4d4802bb144c63620505b42?pvs=4)                 | Wide & Deep Learning for Recommender Systems  |                        
