import streamlit as st
from recomend import recommend_travel, recommend_restaurant

# Streamlit App 제목
st.title("일본 여행지 및 식당 추천 챗봇")
st.caption("사용자의 영화 추천 데이터와 한국 식당 리뷰 데이터를 통해 관광지와 식당을 추천해줍니다.")

def render_message(role, content):
    if role == "assistant":
        # 봇 말풍선 스타일 (그림자, 둥근 모서리, 프로필 이미지 추가)
        st.markdown(
            f"""
            <div style="display: flex; align-items: flex-start; margin-bottom: 10px;">
                <img src="https://img.freepik.com/free-vector/chatbot-chat-message-vectorart_78370-4104.jpg" alt="Bot" 
                     style="width: 30px; height: 30px; border-radius: 50%; margin-right: 10px;">
                <div style="max-width: 70%; padding: 10px; background-color: #e9ecef; color: #000000; 
                            border-radius: 15px; border-top-left-radius: 0px; 
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1); font-family: Arial, sans-serif;">
                    {content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif role == "user":
        # 사용자 말풍선 스타일 (그림자, 둥근 모서리, 프로필 이미지 추가)
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; align-items: flex-start; margin-bottom: 10px;">
                <div style="max-width: 70%; padding: 10px; background-color: #d1e7dd; color: #000000; 
                            border-radius: 15px; border-top-right-radius: 0px; 
                            box-shadow: -2px 2px 5px rgba(0, 0, 0, 0.1); font-family: Arial, sans-serif;">
                    {content}
                </div>
                <img src="https://w7.pngwing.com/pngs/710/71/png-transparent-profle-person-profile-user-circle-icons-icon-thumbnail.png" alt="User" 
                     style="width: 30px; height: 30px; border-radius: 50%; margin-left: 10px;">
            </div>
            """,
            unsafe_allow_html=True,
        )





# 세션 상태 초기화
if "step" not in st.session_state:
    st.session_state["step"] = 0  # 현재 단계
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "travel_preference" not in st.session_state:
    st.session_state["travel_preference"] = None
if "travel_result" not in st.session_state:
    st.session_state["travel_result"] = None
if "restaurant_preference" not in st.session_state:
    st.session_state["restaurant_preference"] = None
if "restaurant_result" not in st.session_state:
    st.session_state["restaurant_result"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "안녕하세요! 사용자 ID를 입력해주세요."}
    ]  # 초기 메시지

# 대화 기록 출력
for message in st.session_state["messages"]:
    render_message(message["role"], message["content"])  # render_message로 변경


# 사용자 입력 처리
if user_input := st.chat_input("메시지를 입력하세요"):
    # 사용자 메시지를 대화에 추가
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # 단계별 처리
    if st.session_state["step"] == 0:  # 사용자 ID 입력 단계
        st.session_state["user_id"] = user_input.strip()
        st.session_state["messages"].append(
            {"role": "assistant", "content": "사용자 ID가 확인되었습니다. 어떤 여행을 하고 싶으신가요? (예: 문화 탐방, 자연 여행 등)"}
        )
        st.session_state["step"] += 1
        st.rerun()  # 즉시 재실행

    elif st.session_state["step"] == 1:  # 여행 선호도 입력 단계
    # 사용자의 입력을 저장
        st.session_state["travel_preference"] = user_input.strip()

        # 사용자 메시지를 한 번만 추가하고 출력
        if len(st.session_state["messages"]) == st.session_state["step"] + 2:  # 메시지가 중복 추가되지 않도록 조건 확인
            st.session_state["messages"].append({"role": "user", "content": user_input.strip()})

        # "여행지를 추천받고 있습니다..." 메시지 추가
        waiting_message = {"role": "assistant", "content": "여행지를 추천받고 있습니다... 잠시만 기다려주세요."}
        st.session_state["messages"].append(waiting_message)

        # 즉시 챗봇 메시지 출력 (render_message 사용)
        render_message("assistant", waiting_message["content"])

        try:
            # 여행지 추천 함수 호출
            travel_result = recommend_travel(
                st.session_state["user_id"], st.session_state["travel_preference"]
            )
            st.session_state["travel_result"] = travel_result

            # 추천 결과 메시지 생성
            travel_description = (
                f"추천 여행지: **{travel_result.get('travel_name', '추천된 여행지 없음')}**\n\n"
                f"{travel_result.get('travel_explanation', '여행지 설명이 없습니다.')}\n\n"
                f"**주소:** {travel_result.get('travel_address', '주소 정보가 없습니다.')}"
            )
            st.session_state["messages"].append({"role": "assistant", "content": travel_description})
            st.session_state["messages"].append(
                {"role": "assistant", "content": "이 여행지가 마음에 드시나요? '네' 또는 '아니오'를 입력해주세요."}
            )
            st.session_state["step"] += 1
        except Exception as e:
            st.session_state["messages"].append(
                {"role": "assistant", "content": f"여행지 추천 중 오류 발생: {e}"}
            )
        st.rerun()  # 즉시 재실행


    elif st.session_state["step"] == 2:  # 여행지 결과 확인 단계
        if user_input.strip().lower() == "네":
            st.session_state["messages"].append(
                {"role": "assistant", "content": "좋습니다! 먹고 싶은 음식이 있나요? 있다면 음식과 원하는 식당 (ex) 가성비 좋은 식당)을 동시에 입력해주세요. 없다면 원하는 식당만 입력해주세요: "}
            )
            st.session_state["step"] += 1
        elif user_input.strip().lower() == "아니오":
            st.session_state["messages"].append(
                {"role": "assistant", "content": "알겠습니다. 다른 여행지를 추천받고 있습니다... 잠시만 기다려주세요."}
            )
            try:
                # 다른 여행지 추천
                travel_result = recommend_travel(
                    st.session_state["user_id"], st.session_state["travel_preference"]
                )
                st.session_state["travel_result"] = travel_result
                travel_description = (
                    f"추천 여행지: **{travel_result.get('travel_name', '추천된 여행지 없음')}**\n\n"
                    f"{travel_result.get('travel_explanation', '여행지 설명이 없습니다.')}\n\n"
                    f"**주소:** {travel_result.get('travel_address', '주소 정보가 없습니다.')}"
                )
                st.session_state["messages"].append(
                    {"role": "assistant", "content": travel_description}
                )
                st.session_state["messages"].append(
                    {"role": "assistant", "content": "이 여행지가 마음에 드시나요? '네' 또는 '아니오'를 입력해주세요."}
                )
            except Exception as e:
                st.session_state["messages"].append(
                    {"role": "assistant", "content": f"여행지 추천 중 오류 발생: {e}"}
                )
        st.rerun()  # 즉시 재실행

    elif st.session_state["step"] == 3:  # 식당 추천 입력 단계
        st.session_state["restaurant_preference"] = user_input.strip()

        # "식당을 추천받고 있습니다..." 메시지 추가
        waiting_message = {"role": "assistant", "content": "식당을 추천받고 있습니다... 잠시만 기다려주세요."}
        st.session_state["messages"].append(waiting_message)

        # 즉시 챗봇 메시지 출력
        render_message("assistant", waiting_message["content"])

        try:
            # 식당 추천 함수 호출
            restaurant_result = recommend_restaurant(
                st.session_state["user_id"],
                st.session_state["restaurant_preference"],
                st.session_state["travel_result"].get("travel_address", ""),
            )
            st.write("DEBUG: Restaurant Recommendation Result", restaurant_result)

            st.session_state["restaurant_result"] = restaurant_result

            # 반환된 데이터 확인 및 메시지 추가
            if "restaurant_explanation" in restaurant_result:
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": f"추천 식당: {restaurant_result['restaurant_explanation']}",
                    }
                )
            elif "error" in restaurant_result:
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": f"오류: {restaurant_result['error']}",
                    }
                )
            st.session_state["messages"].append(
                {"role": "assistant", "content": "이 식당이 마음에 드시나요? '네' 또는 '아니오'를 입력해주세요."}
            )
            st.session_state["step"] += 1  # 다음 단계로 이동
        except Exception as e:
            st.session_state["messages"].append(
                {"role": "assistant", "content": f"식당 추천 중 오류 발생: {e}"}
            )
        st.rerun()  # 즉시 재실행


    elif st.session_state["step"] == 4:  # 식당 추천 결과 확인 단계
        if user_input.strip().lower() == "네":
            st.session_state["messages"].append(
                {"role": "assistant", "content": "좋습니다! 추천을 마칩니다. 즐거운 여행 되세요!"}
            )
            st.session_state["step"] = 0  # 초기화
        elif user_input.strip().lower() == "아니오":
            st.session_state["messages"].append(
                {"role": "assistant", "content": "알겠습니다. 다른 식당을 추천받고 있습니다... 잠시만 기다려주세요."}
            )
            try:
                # 다른 식당 추천
                restaurant_result = recommend_restaurant(
                    st.session_state["user_id"],
                    st.session_state["restaurant_preference"],
                    st.session_state["travel_result"].get("travel_address", ""),
                )
                st.session_state["restaurant_result"] = restaurant_result

                # 반환된 데이터 확인 및 메시지 추가
                if "restaurant_explanation" in restaurant_result:
                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": f"추천 식당: {restaurant_result['restaurant_explanation']}",
                        }
                    )
                elif "error" in restaurant_result:
                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": f"오류: {restaurant_result['error']}",
                        }
                    )
                st.session_state["messages"].append(
                    {"role": "assistant", "content": "이 식당이 마음에 드시나요? '네' 또는 '아니오'를 입력해주세요."}
                )
            except Exception as e:
                st.session_state["messages"].append(
                    {"role": "assistant", "content": f"식당 추천 중 오류 발생: {e}"}
                )
            st.rerun()  # 즉시 재실행

