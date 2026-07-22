import json
import time

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="끝말잇기 챗봇 배틀", page_icon="⏱️", layout="centered")

# ---------------------------------------------------------
# 두음법칙 등가 그룹 (ㄹ/ㄴ ↔ ㅇ 계열 변환 허용)
# ---------------------------------------------------------
EQUIV_PAIRS = [
    ("라", "나"), ("래", "내"), ("러", "너"), ("로", "노"), ("루", "누"), ("르", "느"),
    ("랴", "야"), ("려", "여"), ("례", "예"), ("료", "요"), ("류", "유"), ("리", "이"),
    ("냐", "야"), ("녀", "여"), ("녜", "예"), ("뇨", "요"), ("뉴", "유"), ("니", "이"),
]


def _build_equiv_map(pairs):
    graph = {}
    for a, b in pairs:
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    closure = {}
    for node in graph:
        seen = {node}
        stack = [node]
        while stack:
            cur = stack.pop()
            for nxt in graph.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        closure[node] = seen
    return closure


EQUIV_MAP = _build_equiv_map(EQUIV_PAIRS)


def acceptable_next_chars(last_char):
    return EQUIV_MAP.get(last_char, {last_char})


def check_chain_rule(prev_word, next_word):
    if not prev_word:
        return True, ""
    last_char = prev_word[-1]
    first_char = next_word[0]
    if first_char in acceptable_next_chars(last_char):
        return True, ""
    return False, f"'{prev_word}'의 마지막 글자 '{last_char}'로 시작하는 단어가 아니에요."


# ---------------------------------------------------------
# Solar API 호출 (Upstage, OpenAI 호환 형식)
# ---------------------------------------------------------
def call_solar(api_key, messages, max_tokens=150, model="solar-pro2"):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.8,
    }
    start = time.time()
    resp = requests.post(
        "https://api.upstage.ai/v1/chat/completions",
        headers=headers,
        json=body,
        timeout=30,
    )
    elapsed = time.time() - start
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    # 코드블록(```json ... ```)으로 감싸서 올 때가 있어 벗겨내고 파싱
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    return json.loads(cleaned), elapsed


def validate_user_word(api_key, word, used_words):
    system = (
        "너는 한국어 끝말잇기 게임의 심판이야. 제시된 단어가 "
        "1) 실제로 존재하는 한국어 단어(명사)이고, "
        "2) 이미 사용된 단어 목록에 없는지 판단해. "
        "글자 이어짐 규칙은 이미 확인이 끝났으니 신경 쓰지 마. "
        "다른 설명이나 코드블록 없이 JSON 객체 하나만 출력해: "
        '{"valid": true 또는 false, "reason": "짧은 이유"}'
    )
    user_msg = f"단어: {word}\n이미 사용된 단어들: {', '.join(used_words) if used_words else '없음'}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    result, elapsed = call_solar(api_key, messages, max_tokens=100)
    return result.get("valid", False), result.get("reason", ""), elapsed


DIFFICULTY_GUIDE = {
    "하": "누구나 아는 아주 쉽고 흔한 2글자 단어 위주로 골라줘. 상대를 배려해줘.",
    "중": "일반적으로 널리 쓰이는 단어를 자유롭게 골라줘.",
    "상": (
        "일부러 상대가 잇기 어려운, 흔치 않은 글자로 끝나는 어려운 단어를 골라서 "
        "상대를 최대한 곤란하게 만들어줘. 단, 실제 존재하는 단어여야 해."
    ),
}


def get_ai_word(api_key, used_words, prev_word, difficulty):
    system = (
        "너는 한국어 끝말잇기 게임을 하는 상대야. "
        f"난이도는 '{difficulty}'이고, 지침: {DIFFICULTY_GUIDE[difficulty]} "
        "규칙: 반드시 실제로 존재하는 한국어 명사만 사용하고, "
        "이미 사용된 단어는 다시 쓰면 안 되고, "
        "직전 단어의 마지막 글자(두음법칙 허용)로 시작하는 단어를 제시해야 해. "
        "정말 이어갈 단어가 없으면 포기해도 돼. "
        "다른 설명이나 코드블록 없이 JSON 객체 하나만 출력해: "
        '{"word": "단어" 또는 null, "give_up": true 또는 false}'
    )
    prev_display = prev_word if prev_word else "없음 (자유롭게 아무 단어로나 시작해도 돼)"
    user_msg = (
        f"직전 단어: {prev_display}\n"
        f"이미 사용된 단어들: {', '.join(used_words) if used_words else '없음'}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    result, elapsed = call_solar(api_key, messages, max_tokens=100)
    return result.get("word"), result.get("give_up", False), elapsed


def compute_score(elapsed, time_limit):
    ratio = max(0.0, (time_limit - elapsed) / time_limit)
    return round(ratio * 100)


# ---------------------------------------------------------
# 세션 상태 초기화
# ---------------------------------------------------------
def init_state():
    defaults = {
        "stage": "setup",  # setup / playing / over
        "api_key": "",
        "difficulty": "중",
        "time_limit": 10,
        "first_turn": "나",
        "turn": None,
        "history": [],
        "turn_start": None,
        "winner": None,
        "reason": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

st.title("⏱️ 끝말잇기 챗봇 배틀")

# ---------------------------------------------------------
# 1) 셋업 화면
# ---------------------------------------------------------
if st.session_state.stage == "setup":
    st.caption("Solar와 실시간 끝말잇기 대결! 느리면 점수가 깎이고, 시간 초과되면 바로 패배해요.")
    st.caption("(Solar API Key는 앱에 미리 설정되어 있어서 따로 입력할 필요 없어요.)")

    api_key = st.secrets.get("SOLAR_API_KEY", "")
    if not api_key:
        st.error(
            "Solar API Key가 설정되어 있지 않아요. "
            "Streamlit Cloud의 앱 설정(Settings) → Secrets에 다음처럼 추가해주세요:\n\n"
            'SOLAR_API_KEY = "up_..."'
        )

    with st.form("setup_form"):
        difficulty = st.radio("난이도", ["하", "중", "상"], index=1, horizontal=True)
        time_limit = st.radio("턴당 제한시간(초)", [3, 5, 10, 30], index=2, horizontal=True)
        first_turn = st.radio("먼저 시작할 사람", ["나", "챗봇"], horizontal=True)
        submitted = st.form_submit_button("게임 시작", use_container_width=True, disabled=not api_key)

    if submitted and api_key:
        st.session_state.api_key = api_key
        st.session_state.difficulty = difficulty
        st.session_state.time_limit = time_limit
        st.session_state.first_turn = first_turn
        st.session_state.turn = first_turn
        st.session_state.history = []
        st.session_state.winner = None
        st.session_state.reason = ""
        st.session_state.turn_start = time.time()
        st.session_state.stage = "playing"
        st.rerun()

# ---------------------------------------------------------
# 2) 플레이 화면
# ---------------------------------------------------------
elif st.session_state.stage == "playing":
    my_total = sum(h["score"] for h in st.session_state.history if h["by"] == "나")
    ai_total = sum(h["score"] for h in st.session_state.history if h["by"] == "챗봇")
    col1, col2 = st.columns(2)
    col1.metric("🙋 나의 점수", my_total)
    col2.metric("🤖 챗봇 점수", ai_total)

    prev_word = st.session_state.history[-1]["word"] if st.session_state.history else None
    if prev_word:
        st.markdown(f"### 현재 단어: **{prev_word}**")
    else:
        st.markdown("### 첫 단어를 자유롭게 시작하세요!")

    used_words = [h["word"] for h in st.session_state.history]

    if st.session_state.turn == "나":
        st_autorefresh(interval=300, key="user_timer_refresh")

        elapsed_so_far = time.time() - st.session_state.turn_start
        remaining = max(0.0, st.session_state.time_limit - elapsed_so_far)
        st.progress(min(1.0, remaining / st.session_state.time_limit))
        st.markdown(f"#### 남은 시간: {remaining:0.1f}초")

        if remaining <= 0:
            st.session_state.winner = "챗봇"
            st.session_state.reason = "시간 초과로 패배했어요."
            st.session_state.stage = "over"
            st.rerun()

        with st.form("word_form", clear_on_submit=True):
            word = st.text_input("단어 입력", key="word_input")
            submitted = st.form_submit_button("제출")

        if submitted and word.strip():
            elapsed = time.time() - st.session_state.turn_start
            word = word.strip()

            if elapsed > st.session_state.time_limit:
                st.session_state.winner = "챗봇"
                st.session_state.reason = "시간 초과로 패배했어요."
                st.session_state.stage = "over"
                st.rerun()

            ok, msg = check_chain_rule(prev_word, word)
            if not ok:
                st.error(msg)
            elif word in used_words:
                st.error("이미 사용한 단어예요.")
            else:
                with st.spinner("단어 확인 중..."):
                    try:
                        valid, reason, _ = validate_user_word(st.session_state.api_key, word, used_words)
                    except Exception as e:
                        st.error(f"API 호출 중 오류가 발생했어요: {e}")
                        valid = None
                if valid is False:
                    st.error(f"유효하지 않은 단어예요: {reason}")
                elif valid is True:
                    score = compute_score(elapsed, st.session_state.time_limit)
                    st.session_state.history.append(
                        {"word": word, "by": "나", "elapsed": round(elapsed, 2), "score": score}
                    )
                    st.session_state.turn = "챗봇"
                    st.session_state.turn_start = time.time()
                    st.rerun()

    else:  # 챗봇 턴
        st.markdown("#### 🤖 챗봇이 답변을 준비하고 있어요...")
        try:
            with st.spinner("챗봇이 생각 중..."):
                word, give_up, elapsed = get_ai_word(
                    st.session_state.api_key, used_words, prev_word, st.session_state.difficulty
                )
        except Exception as e:
            st.error(f"API 호출 중 오류가 발생했어요: {e}")
            st.stop()

        if give_up or not word:
            st.session_state.winner = "나"
            st.session_state.reason = "챗봇이 이어갈 단어를 찾지 못해 패배했어요."
            st.session_state.stage = "over"
            st.rerun()

        ok, _ = check_chain_rule(prev_word, word)
        if elapsed > st.session_state.time_limit or not ok or word in used_words:
            if elapsed > st.session_state.time_limit:
                reason = "챗봇이 제한시간을 넘겨서 패배했어요."
            else:
                reason = "챗봇이 규칙에 맞지 않는 단어를 내서 패배했어요."
            st.session_state.winner = "나"
            st.session_state.reason = reason
            st.session_state.stage = "over"
            st.rerun()

        score = compute_score(elapsed, st.session_state.time_limit)
        st.session_state.history.append(
            {"word": word, "by": "챗봇", "elapsed": round(elapsed, 2), "score": score}
        )
        st.session_state.turn = "나"
        st.session_state.turn_start = time.time()
        st.rerun()

    if st.session_state.history:
        st.markdown("---")
        st.markdown("#### 진행 기록")
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df[["by", "word", "elapsed", "score"]], use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# 3) 게임 종료 화면
# ---------------------------------------------------------
else:
    st.markdown(f"### 🏁 승자: {'🙋 나' if st.session_state.winner == '나' else '🤖 챗봇'}")
    st.info(st.session_state.reason)

    my_total = sum(h["score"] for h in st.session_state.history if h["by"] == "나")
    ai_total = sum(h["score"] for h in st.session_state.history if h["by"] == "챗봇")
    col1, col2 = st.columns(2)
    col1.metric("🙋 나의 최종 점수", my_total)
    col2.metric("🤖 챗봇 최종 점수", ai_total)

    if st.session_state.history:
        st.markdown("#### 전체 기록")
        df = pd.DataFrame(st.session_state.history)
        df.insert(0, "라운드", df.index + 1)
        st.dataframe(
            df.rename(columns={"by": "플레이어", "word": "단어", "elapsed": "소요시간(초)", "score": "점수"}),
            use_container_width=True,
            hide_index=True,
        )

    if st.button("다시 시작", use_container_width=True):
        for k in ["stage", "history", "turn", "turn_start", "winner", "reason"]:
            st.session_state.pop(k, None)
        st.rerun()
