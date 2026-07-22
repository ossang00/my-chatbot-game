import json
import time

import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="끝말잇기 챗봇 배틀", page_icon="⏱️", layout="centered")

# ---------------------------------------------------------
# 두음법칙 (ㄹ→ㄴ/ㅇ, ㄴ→ㅇ) - 받침 유무와 상관없이 모든 음절에 적용
# ---------------------------------------------------------
CHOSUNG_LIST = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
JUNGSUNG_LIST = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]
# ㅑ, ㅕ, ㅖ, ㅛ, ㅠ, ㅣ 처럼 반모음 'ㅣ' 계열이 낀 모음 (두음법칙에서 ㄹ이 아예 탈락하는 경우)
Y_GLIDE_JUNG_INDICES = {2, 6, 7, 12, 17, 20}
CHO_RIEUL = CHOSUNG_LIST.index("ㄹ")
CHO_NIEUN = CHOSUNG_LIST.index("ㄴ")
CHO_IEUNG = CHOSUNG_LIST.index("ㅇ")


def _decompose_syllable(ch):
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return None
    cho = code // (21 * 28)
    jung = (code % (21 * 28)) // 28
    jong = code % 28
    return cho, jung, jong


def _compose_syllable(cho, jung, jong):
    return chr(0xAC00 + (cho * 21 + jung) * 28 + jong)


def acceptable_next_chars(last_char):
    """직전 단어의 마지막 글자를 기준으로, 다음 단어가 시작할 수 있는 글자들을 계산한다.
    받침이 있든 없든 모든 음절에 두음법칙(ㄹ→ㄴ/ㅇ, ㄴ→ㅇ)을 일반적으로 적용한다.
    예: 람→남, 력→역, 랑→낭, 리→이 등."""
    decomposed = _decompose_syllable(last_char)
    if decomposed is None:
        return {last_char}

    cho, jung, jong = decomposed
    result = {last_char}

    if cho == CHO_RIEUL:
        alt_cho = CHO_IEUNG if jung in Y_GLIDE_JUNG_INDICES else CHO_NIEUN
        result.add(_compose_syllable(alt_cho, jung, jong))
    elif cho == CHO_NIEUN and jung in Y_GLIDE_JUNG_INDICES:
        result.add(_compose_syllable(CHO_IEUNG, jung, jong))

    return result


def check_chain_rule(prev_word, next_word):
    if not prev_word:
        return True, ""
    last_char = prev_word[-1]
    first_char = next_word[0]
    if first_char in acceptable_next_chars(last_char):
        return True, ""
    return False, f"'{prev_word}'의 마지막 글자 '{last_char}'로 시작하는 단어가 아니에요."


def is_pure_hangul_word(word):
    """완성형 한글 음절로만 이루어진 단어인지 확인한다 (특수문자·영문·숫자·공백 등은 불허)."""
    if not word:
        return False
    return all("가" <= ch <= "힣" for ch in word)


# ---------------------------------------------------------
# Solar API 호출 (Upstage, OpenAI 호환 형식)
# ---------------------------------------------------------
def _call_solar_once(api_key, messages, max_tokens=150, model="solar-pro2", temperature=0.8):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
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


def call_solar(api_key, messages, max_tokens=150, model="solar-pro2", temperature=0.8):
    """JSON 파싱이 실패하면 한 번 더 자동으로 재시도한다."""
    last_error = None
    for _ in range(2):
        try:
            return _call_solar_once(
                api_key, messages, max_tokens=max_tokens, model=model, temperature=temperature
            )
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            continue
    raise last_error


def _validate_user_word_once(api_key, word):
    system = (
        "너는 한국어 어휘에 정통한 심판이야. 표준국어대사전 기준으로, 제시된 단어가 "
        "실제로 존재하는 한국어 단어(명사)인지만 판단해. "
        "다음과 같은 유형도 전부 명백히 유효한 단어로 취급해: "
        "구체명사(사과, 책상), 추상명사(회한, 슬픔, 자유, 우정, 인내), 감정·심리 관련 명사, "
        "전문용어·학술용어(어류, 양서류, 광합성), 한자어, 외래어, 고유명사(지명·인명이 아닌 일반적 고유명사), "
        "옛말이 아닌 이상 다소 예스럽거나 문어체적인 단어. "
        "네가 즉시 뜻이 떠오르지 않더라도, 실제 한국어에 존재할 법한 단어라면 무효로 판단하지 마. "
        "명백히 오타이거나, 존재하지 않는 조합이거나, 명사가 아닌 동사·형용사 활용형일 때만 무효로 판단해. "
        "특히 '슬픈', '예쁜', '빠른', '먹은', '좋은'처럼 형용사·동사에 '-ㄴ/-은/-는'이 붙어 뒷말을 꾸미는 "
        "관형사형은 명사가 아니므로 무효야. 대신 그에 대응하는 명사형(슬픔, 아름다움, 속도, 식사, 장점 등)은 유효해. "
        "이미 사용된 단어인지 여부나 글자 이어짐 규칙은 이미 다른 곳에서 확인이 끝났으니 신경 쓰지 마. "
        "다른 설명이나 코드블록 없이 JSON 객체 하나만 출력해: "
        '{"valid": true 또는 false, "reason": "짧은 이유"}'
    )
    user_msg = f"단어: {word}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    result, elapsed = call_solar(api_key, messages, max_tokens=100, temperature=0.1)
    return result.get("valid", False), result.get("reason", ""), elapsed


def validate_user_word(api_key, word, max_checks=3):
    """무효 판정이 나오면, 실수(할루시네이션)일 수 있으니 재확인한다.
    여러 번 물어봐서 단 한 번이라도 유효하다고 하면 유효로 인정하고,
    max_checks번 모두 무효라고 해야만 최종적으로 무효 처리한다."""
    total_elapsed = 0.0
    last_reason = ""
    for _ in range(max_checks):
        valid, reason, elapsed = _validate_user_word_once(api_key, word)
        total_elapsed += elapsed
        last_reason = reason
        if valid:
            return True, reason, total_elapsed
    return False, last_reason, total_elapsed


DIFFICULTY_GUIDE = {
    "하": (
        "유치원생~초등학교 저학년도 바로 알 수 있는 아주 쉽고 흔한 2글자 명사만 사용해 "
        "(예: 사과, 바나나, 강아지, 고양이, 학교, 나무, 하늘, 구름 같은 수준). "
        "사자성어, 한자어, 전문용어, 3글자 이상의 단어는 절대 쓰지 마. "
        "그리고 일부러 상대가 잇기 쉬운, 흔한 글자로 끝나는 단어를 골라줘. "
        "같은 단어만 계속 반복하지는 말고, 쉬운 단어 안에서 다양하게 골라줘."
    ),
    "중": (
        "3~4글자의 일반 성인이 아는 단어를 폭넓게 사용해. 사자성어, 전문 용어, 지명, 학술 용어도 섞어서 "
        "상대가 방심하지 않게 만들어줘."
    ),
    "상": (
        "너는 끝말잇기 최상급 고수야. 목표는 상대를 반드시 이기는 것. "
        "단순히 어려운 단어를 내는 게 아니라, '이 글자로 시작하는 단어가 거의 없는' 궁지의 글자로 "
        "끝나는 단어를 전략적으로 골라야 해. "
        "한국어에는 '한방단어'라 불리는, 특정 음절로 끝나서 상대가 절대 이어갈 수 없게 만드는 "
        "유명한 단어들이 있어. 대표적인 패턴은: "
        "① 화학 원소명 등 외래어 계열 어미(-늄, -륨, -튬, -퓸처럼 원소 이름 끝음절), "
        "② 외래어/전문용어의 마지막 음절(-츠, -톤, -폰, -크 등 받침 없는 외래어 어미), "
        "③ 겹받침(ㄶ/ㄺ/ㄼ/ㅄ 등)이나 흔치 않은 겹모음(ㅢ/ㅘ/ㅝ/ㅟ 등)으로 끝나는 단어, "
        "④ 두음법칙조차 통하지 않는 음절(해당 음절로 시작하는 표준어 명사가 사실상 없는 경우). "
        "이런 패턴에 해당하는 단어를 최우선으로 사용해서 상대를 궁지에 몰아넣어. "
        "여러 후보가 떠오르면 그중 가장 상대를 궁지에 몰 수 있는 단어를 선택해. "
        "일부러 봐주거나 쉬운 단어로 물러서지 마. 반드시 실제 존재하는 단어여야 하고, "
        "3~6글자 정도의 단어를 적극적으로 사용해도 좋아. "
        "반대로 상대가 너에게 이런 '한방단어'를 냈을 때는 절대 곧바로 포기하지 말고, "
        "다음 순서로 최대한 뒤져봐: "
        "(1) 그 음절의 두음법칙 변환형(ㄹ→ㄴ/ㅇ, ㄴ→ㅇ)으로 시작하는 단어가 있는지, "
        "(2) 표준어 외에 방언·옛말·북한어·전문용어·외래어 중에 그 음절로 시작하는 단어가 있는지, "
        "(3) 그래도 정말 없다면, 이건 실제로 한국어에 존재하지 않는다는 뜻이니 억지로 지어내지 말고 "
        "정직하게 포기해도 괜찮아. 이런 경우는 실력 부족이 아니라 그 음절 자체가 원래 대응이 "
        "거의 불가능한 유명한 필살 음절이라서 그런 거야."
    ),
}


def get_ai_word(api_key, used_words, prev_word, difficulty, rejected_words=None):
    expected_chars = acceptable_next_chars(prev_word[-1]) if prev_word else None
    is_opening_move = prev_word is None

    if is_opening_move:
        # 챗봇이 게임의 첫 단어를 낼 때는 상대가 아직 한 턴도 안 한 상태이므로,
        # 난이도(특히 '상'의 공격적인 전략)와 상관없이 무난하고 평범한 단어로 시작한다.
        difficulty_instruction = (
            "지금은 게임의 아주 첫 단어를 네가 먼저 내는 상황이야. 상대는 아직 한 턴도 하지 않았으니 "
            "'한방단어'(알루미늄, 늄으로 끝나는 단어, 겹받침 등 상대를 바로 궁지에 모는 단어)는 "
            "절대 쓰지 마. 누구나 쉽게 이어갈 수 있는 평범하고 무난한 2~3글자 명사로 시작해."
        )
    else:
        difficulty_instruction = f"난이도는 '{difficulty}'이고, 지침: {DIFFICULTY_GUIDE[difficulty]}"

    system = (
        "너는 한국어 끝말잇기 게임을 하는 상대야. "
        f"{difficulty_instruction} "
        "규칙: 반드시 실제로 존재하는 한국어 '명사'만 사용하고, "
        "형용사나 동사의 활용형(관형사형 포함)은 명사가 아니니 절대 내면 안 돼. "
        "예를 들어 '슬픈', '예쁜', '빠른', '먹은'은 명사가 아니라 형용사·동사 활용형이라 무효야. "
        "대신 그 뜻에 해당하는 명사형(예: 슬픔, 아름다움, 속도, 식사)을 사용해. "
        "단어는 최소 2글자 이상이어야 하고(한 글자짜리는 안 돼), "
        "이미 사용된 단어는 다시 쓰면 안 되고, "
        "직전 단어의 마지막 글자(두음법칙 허용)로 시작하는 단어를 제시해야 해. "
        "규칙을 지키는 게 최우선이니 제시하기 전에 스스로 다시 한번 확인해. "
        "정말로 이어갈 단어가 하나도 없을 때만 포기해도 돼. "
        "다른 설명이나 코드블록 없이 JSON 객체 하나만 출력해: "
        '{"word": "단어" 또는 null, "give_up": true 또는 false}'
    )
    prev_display = prev_word if prev_word else "없음 (자유롭게 아무 단어로나 시작해도 돼)"
    user_msg = (
        f"직전 단어: {prev_display}\n"
        f"이미 사용된 단어들: {', '.join(used_words) if used_words else '없음'}"
    )
    if expected_chars:
        user_msg += f"\n다음 단어는 반드시 이 글자들 중 하나로 시작해야 해: {', '.join(sorted(expected_chars))}"
    if rejected_words:
        user_msg += (
            f"\n다음 단어들은 방금 규칙 위반이나 오류로 거절됐어. 다시 내지 마: "
            f"{', '.join(rejected_words)}"
        )

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    result, elapsed = call_solar(api_key, messages, max_tokens=100)
    return result.get("word"), result.get("give_up", False), elapsed


def get_ai_word_with_retries(api_key, used_words, prev_word, difficulty, time_limit, max_attempts=3):
    """챗봇이 규칙 위반이나 실수로 곧바로 지지 않도록, 정말 포기하기 전까지 몇 번 더 시도해본다.
    반환값: (word 또는 None, give_up 여부, 총 걸린 시간)"""
    rejected = []
    total_elapsed = 0.0

    for attempt in range(max_attempts):
        word, give_up, elapsed = get_ai_word(
            api_key, used_words, prev_word, difficulty, rejected_words=rejected
        )
        total_elapsed += elapsed

        if give_up or not word:
            return None, True, total_elapsed

        word = word.strip()
        ok = check_chain_rule(prev_word, word)[0] if prev_word else True
        is_long_enough = len(word) >= 2
        is_hangul = is_pure_hangul_word(word)

        if ok and is_long_enough and is_hangul and word not in used_words:
            # 명사가 맞는지(형용사·동사 활용형이 아닌지) 한 번 더 확인한다.
            try:
                is_noun, _, noun_check_elapsed = validate_user_word(api_key, word, max_checks=1)
                total_elapsed += noun_check_elapsed
            except Exception:
                is_noun = True  # 검증 자체가 실패하면(네트워크 등) 통과시켜준다.
            if is_noun:
                return word, False, total_elapsed

        # 규칙 위반, 명사가 아니거나, 중복 단어면 실패로 기록하고 다시 시도
        rejected.append(word)
        if total_elapsed > time_limit:
            break

    # 정해진 횟수를 다 써도 유효한 단어를 못 찾으면 포기한 것으로 처리
    return None, True, total_elapsed


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

        error_slot = st.empty()

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
            if not is_pure_hangul_word(word):
                error_slot.error("한글 단어만 입력할 수 있어요. (특수문자·영문·숫자·띄어쓰기 불가)")
            elif not ok:
                error_slot.error(msg)
            elif word in used_words:
                error_slot.error("이미 사용한 단어예요.")
            else:
                with st.spinner("단어 확인 중..."):
                    try:
                        valid, reason, _ = validate_user_word(st.session_state.api_key, word)
                    except Exception as e:
                        error_slot.error(f"API 호출 중 오류가 발생했어요: {e}")
                        valid = None
                if valid is False:
                    error_slot.error(f"유효하지 않은 단어예요: {reason}")
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
                    word, give_up, api_elapsed = get_ai_word_with_retries(
                        st.session_state.api_key,
                        used_words,
                        prev_word,
                        st.session_state.difficulty,
                        st.session_state.time_limit,
                        max_attempts=5 if st.session_state.difficulty == "상" else 3,
                    )
            except Exception as e:
                st.error(f"API 호출 중 오류가 발생했어요: {e}")
                st.stop()

            is_loss = give_up or not word or api_elapsed > st.session_state.time_limit

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
    if st.session_state.history:
        st.markdown("#### 전체 기록")
        for h in st.session_state.history:
            role = "user" if h["by"] == "나" else "assistant"
            avatar = "🙋" if h["by"] == "나" else "🤖"
            with st.chat_message(role, avatar=avatar):
                st.markdown(f"**{h['word']}**")

    st.markdown(f"### 🏁 승자: {'🙋 나' if st.session_state.winner == '나' else '🤖 챗봇'}")
    st.info(st.session_state.reason)

    if st.button("다시 시작", use_container_width=True):
        keys_to_clear = [
            "stage", "history", "turn", "turn_start", "winner", "reason",
            "ai_call_done", "ai_result_word", "ai_result_is_loss", "timeout_result",
        ]
        for k in keys_to_clear:
            st.session_state.pop(k, None)
        st.rerun()
