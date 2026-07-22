import json
import time

import requests
import streamlit as st
import streamlit.components.v1 as components

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
def _call_solar_once(api_key, messages, max_tokens=150, model="solar-pro2"):
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
    # 응답 앞뒤로 부연설명이나 여분의 텍스트가 붙어와도, 첫 JSON 객체 하나만 읽어들인다.
    start_idx = cleaned.find("{")
    if start_idx == -1:
        raise ValueError(f"응답에서 JSON을 찾을 수 없어요: {cleaned!r}")
    obj, _ = json.JSONDecoder().raw_decode(cleaned[start_idx:])
    return obj, elapsed


def call_solar(api_key, messages, max_tokens=150, model="solar-pro2"):
    """JSON 파싱이 실패하면 한 번 더 자동으로 재시도한다."""
    last_error = None
    for _ in range(2):
        try:
            return _call_solar_once(api_key, messages, max_tokens=max_tokens, model=model)
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            continue
    raise last_error


def validate_user_word(api_key, word):
    system = (
        "너는 한국어 끝말잇기 게임의 심판이야. 제시된 단어가 "
        "실제로 존재하는 한국어 단어(명사)인지만 판단해. "
        "이미 사용된 단어인지 여부나 글자 이어짐 규칙은 이미 다른 곳에서 확인이 끝났으니 신경 쓰지 마. "
        "다른 설명이나 코드블록 없이 JSON 객체 하나만 출력해: "
        '{"valid": true 또는 false, "reason": "짧은 이유"}'
    )
    user_msg = f"단어: {word}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    result, elapsed = call_solar(api_key, messages, max_tokens=100)
    return result.get("valid", False), result.get("reason", ""), elapsed


DIFFICULTY_GUIDE = {
    "하": (
        "일상적으로 쓰이는 2~3글자 명사를 골라줘. 너무 유치하거나 매턴 똑같은 단어(예: 사과, 바나나 같은 뻔한 단어) "
        "반복은 피하고, 초등학생도 알 만한 수준에서 다양하게 골라줘."
    ),
    "중": (
        "3~4글자의 일반 성인이 아는 단어를 폭넓게 사용해. 사자성어, 전문 용어, 지명, 학술 용어도 섞어서 "
        "상대가 방심하지 않게 만들어줘."
    ),
    "상": (
        "3~5글자의 어렵고 희귀한 단어(전문 용어, 고사성어, 한자어, 잘 안 쓰이는 명사 등)를 적극적으로 사용해. "
        "특히 'ㅔ, ㅢ, ㅑ, ㅘ' 등 다음 사람이 잇기 힘든 글자로 끝나는 단어를 우선적으로 골라서 "
        "상대를 최대한 궁지에 몰아넣어줘. 절대 쉬운 단어를 봐주지 마. 단, 실제 존재하는 단어여야 해."
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


def autofocus_word_input():
    """단어 입력창에 커서를 옮겨준다 (턴이 바뀔 때 한 번만 호출됨)."""
    components.html(
        """
        <script>
        setTimeout(function() {
            const doc = window.parent.document;
            const inputs = doc.querySelectorAll('input[type="text"]');
            if (inputs.length > 0) {
                inputs[inputs.length - 1].focus();
            }
        }, 80);
        </script>
        """,
        height=0,
    )


@st.fragment(run_every=0.3)
def render_timer():
    """제한시간 표시 전용 fragment. 이 부분만 0.3초마다 갱신되고,
    입력 폼이 있는 나머지 화면은 영향받지 않는다.
    시간이 다 되면 st.session_state.timeout_result 에 미리 저장해둔
    {"winner": ..., "reason": ...} 값으로 게임을 종료시킨다."""
    elapsed_so_far = time.time() - st.session_state.turn_start
    remaining = max(0.0, st.session_state.time_limit - elapsed_so_far)
    st.progress(min(1.0, remaining / st.session_state.time_limit))
    st.markdown(f"#### 남은 시간: {remaining:0.1f}초")

    if remaining <= 0:
        result = st.session_state.get(
            "timeout_result", {"winner": "챗봇", "reason": "시간 초과로 패배했어요."}
        )
        st.session_state.winner = result["winner"]
        st.session_state.reason = result["reason"]
        st.session_state.stage = "over"
        st.rerun()


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
    st.caption("실시간 끝말잇기 대결! 느리면 점수가 깎이고, 시간 초과되면 바로 패배해요.")

    api_key = st.secrets.get("SOLAR_API_KEY", "")
    if not api_key:
        st.error(
            "Solar API Key가 설정되어 있지 않아요. "
            "Streamlit Cloud의 앱 설정(Settings) → Secrets에 다음처럼 추가해주세요:\n\n"
            'SOLAR_API_KEY = "up_..."'
        )

    with st.form("setup_form"):
        difficulty = st.radio("난이도", ["하", "중", "상"], index=1, horizontal=True)
        time_limit = st.radio("턴당 제한시간(초)", [5, 10, 30], index=1, horizontal=True)
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
    prev_word = st.session_state.history[-1]["word"] if st.session_state.history else None
    used_words = [h["word"] for h in st.session_state.history]

    if not st.session_state.history:
        st.markdown("### 첫 단어를 자유롭게 시작하세요!")

    for h in st.session_state.history:
        role = "user" if h["by"] == "나" else "assistant"
        avatar = "🙋" if h["by"] == "나" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(f"**{h['word']}**")

    if st.session_state.turn == "나":
        is_free_first_move = (
            not st.session_state.history and st.session_state.first_turn == "나"
        )

        if is_free_first_move:
            st.info("첫 단어는 시간 제한 없이 자유롭게 입력하세요.")
        else:
            st.session_state.timeout_result = {
                "winner": "챗봇",
                "reason": "시간 초과로 패배했어요.",
            }
            render_timer()

        with st.form("word_form", clear_on_submit=True):
            word = st.text_input("단어 입력", key="word_input")
            submitted = st.form_submit_button("제출")

        if st.session_state.get("_last_focus_ts") != st.session_state.turn_start:
            autofocus_word_input()
            st.session_state._last_focus_ts = st.session_state.turn_start

        if submitted and word.strip():
            elapsed = time.time() - st.session_state.turn_start
            word = word.strip()

            if not is_free_first_move and elapsed > st.session_state.time_limit:
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
                        valid, reason, _ = validate_user_word(st.session_state.api_key, word)
                    except Exception as e:
                        st.error(f"API 호출 중 오류가 발생했어요: {e}")
                        valid = None
                if valid is False:
                    st.error(f"유효하지 않은 단어예요: {reason}")
                elif valid is True:
                    st.session_state.history.append({"word": word, "by": "나"})
                    st.session_state.turn = "챗봇"
                    st.session_state.turn_start = time.time()
                    st.rerun()

    else:  # 챗봇 턴
        if not st.session_state.get("ai_call_done", False):
            st.markdown("#### 🤖 챗봇이 답변을 준비하고 있어요...")
            try:
                with st.spinner("챗봇이 생각 중..."):
                    word, give_up, api_elapsed = get_ai_word(
                        st.session_state.api_key, used_words, prev_word, st.session_state.difficulty
                    )
            except Exception as e:
                st.error(f"API 호출 중 오류가 발생했어요: {e}")
                st.stop()

            if word:
                ok, _ = check_chain_rule(prev_word, word)
            else:
                ok = False

            is_loss = (
                give_up
                or not word
                or api_elapsed > st.session_state.time_limit
                or not ok
                or word in used_words
            )

            st.session_state.ai_call_done = True
            st.session_state.ai_result_word = word
            st.session_state.ai_result_is_loss = is_loss
            st.rerun()

        elif st.session_state.ai_result_is_loss:
            # 챗봇이 사실상 진 상황이어도, 실제로 제한시간이 다 지날 때까지는
            # 타이머가 끝까지 흐르도록 보여준 뒤에 패배 처리한다.
            st.markdown("#### 🤖 챗봇이 고민하고 있어요...")
            st.session_state.timeout_result = {
                "winner": "나",
                "reason": "챗봇이 제한시간을 넘겨서 패배했어요.",
            }
            render_timer()

        else:
            st.session_state.history.append({"word": st.session_state.ai_result_word, "by": "챗봇"})
            st.session_state.turn = "나"
            st.session_state.turn_start = time.time()
            for k in ("ai_call_done", "ai_result_word", "ai_result_is_loss"):
                st.session_state.pop(k, None)
            st.rerun()

# ---------------------------------------------------------
# 3) 게임 종료 화면
# ---------------------------------------------------------
else:
    st.markdown(f"### 🏁 승자: {'🙋 나' if st.session_state.winner == '나' else '🤖 챗봇'}")
    st.info(st.session_state.reason)

    if st.session_state.history:
        st.markdown("#### 전체 기록")
        for h in st.session_state.history:
            role = "user" if h["by"] == "나" else "assistant"
            avatar = "🙋" if h["by"] == "나" else "🤖"
            with st.chat_message(role, avatar=avatar):
                st.markdown(f"**{h['word']}**")

    if st.button("다시 시작", use_container_width=True):
        keys_to_clear = [
            "stage", "history", "turn", "turn_start", "winner", "reason",
            "ai_call_done", "ai_result_word", "ai_result_is_loss", "timeout_result",
        ]
        for k in keys_to_clear:
            st.session_state.pop(k, None)
        st.rerun()
