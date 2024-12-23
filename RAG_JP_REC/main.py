import sys
# from recommendation import recommend_travel, recommend_restaurant
from recomend import recommend_travel, recommend_restaurant

# 프로그램 실행 흐름
def main():
    print("일본 여행 및 식당 추천 프로그램에 오신 것을 환영합니다!")
    
    user_id = input("당신의 사용자 ID를 입력해주세요: ")
    travel_preference = input("어떤 여행을 하고 싶으신가요? : ")
    
    # 여행지 추천
    final_location = recommend_travel(user_id, travel_preference)
    # print(final_location)
    if final_location:
        restaurant_preference = input("먹고 싶은 음식이 있나요? 있다면 원하는 식당 (예: 가성비 좋은 식당, 초밥)을 동시에 입력해주세요: ").strip()

        recommend_restaurant(user_id, restaurant_preference, final_location)

if __name__ == "__main__":
    main()