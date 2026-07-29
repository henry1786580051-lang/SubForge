import difflib
import re
from typing import Any, List, Tuple

from ..llm import call_llm, get_response_text
from ..prompts import get_prompt
from ..utils.logger import setup_logger
from ..utils.text_utils import count_words, is_mainly_cjk
from .boundary import assess_english_boundary
from .length_policy import (
    DEFAULT_CJK_HARD_LIMIT,
    DEFAULT_ENGLISH_SOFT_LIMIT,
    resolve_length_policy,
)

logger = setup_logger("split_by_llm")

MAX_STEPS = 3  # Agent loop max retry count

_DANGLING_ENGLISH_TAILS = {
    "a",
    "an",
    "and",
    "but",
    "into",
    "of",
    "or",
    "the",
}
_DANGLING_ENGLISH_PHRASES = (
    "as much as",
    "because of",
    "idea of",
    "one of",
)


def _has_dangling_english_tail(text: str, following: str = "") -> bool:
    """Return whether a split leaves an obviously incomplete English tail."""
    raw = str(text or "").strip()
    if not raw or re.search(r"[.!?][\"')\]]*$", raw):
        return False
    words = re.findall(r"[A-Za-z0-9']+", raw.lower())
    if not words:
        return False
    if following and assess_english_boundary(raw, following).unstable:
        return True
    if words[-1] in _DANGLING_ENGLISH_TAILS:
        return True
    normalized = " ".join(words)
    return any(normalized.endswith(phrase) for phrase in _DANGLING_ENGLISH_PHRASES)


def split_by_llm(
    text: str,
    model: str = "gpt-4o-mini",
    max_word_count_cjk: int = DEFAULT_CJK_HARD_LIMIT,
    max_word_count_english: int = DEFAULT_ENGLISH_SOFT_LIMIT,
    hard_max_word_count_english: int | None = None,
    llm_client: Any = None,
) -> List[str]:
    """使用LLM进行文本断句（固定使用句子Segments）

    Args:
        text: 待断句的文本
        model: LLM模型名称
        max_word_count_cjk: CJK 硬上限字符数
        max_word_count_english: 英文目标单词数（软限制）
        hard_max_word_count_english: 英文硬上限；省略时使用统一策略

    Returns:
        断句后的文本列表
    """
    try:
        policy = resolve_length_policy(max_word_count_cjk, max_word_count_english)
        return _split_with_agent_loop(
            text,
            model,
            policy.cjk_hard_limit,
            policy.english_soft_limit,
            hard_max_word_count_english or policy.english_hard_limit,
            llm_client,
        )
    except Exception as e:
        logger.error(f"Sentence splitting failed: {e}")
        raise


def _split_with_agent_loop(
    text: str,
    model: str,
    max_word_count_cjk: int,
    max_word_count_english: int,
    hard_max_word_count_english: int,
    llm_client: Any = None,
) -> List[str]:
    """使用agent loop 建立反馈循环进行文本断句，自动验证和修正"""
    prompt_path = "split/sentence"
    system_prompt = get_prompt(
        prompt_path,
        max_word_count_cjk=max_word_count_cjk,
        max_word_count_english=max_word_count_english,
        hard_max_word_count_english=hard_max_word_count_english,
    )

    user_prompt = (
        f"Please use multiple <br> tags to separate the following sentence:\n{text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for step in range(MAX_STEPS):
        response = call_llm(
            messages=messages,
            model=model,
            temperature=0.1,
            client=llm_client,
        )

        result_text = get_response_text(response)

        split_result = _parse_split_response(result_text)

        # 验证结果
        is_valid, error_message = _validate_split_result(
            original_text=text,
            split_result=split_result,
            max_word_count_cjk=max_word_count_cjk,
            max_word_count_english=max_word_count_english,
            hard_max_word_count_english=hard_max_word_count_english,
        )

        if is_valid:
            return split_result

        # 添加反馈到对话
        logger.warning(
            f"Split validation failed. Feedback loop (第{step + 1}次尝试):\n {error_message}\n\n"
        )
        messages.append({"role": "assistant", "content": result_text})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Error: {error_message}\n"
                    "Output ONLY the original text with <br> tags inserted. "
                    "No explanation, no comments, no labels, no markdown. "
                    "The output must contain exactly the original words in the original order."
                ),
            }
        )

    raise RuntimeError(
        "LLM split result failed validation after retries; falling back to rule-based split"
    )


def _parse_split_response(result_text: str) -> List[str]:
    """Parse LLM split output and remove common wrapper noise.

    Some OpenAI-compatible models prepend explanations during the feedback loop
    ("I fixed the long segments..."). Returning that text poisons downstream
    word-timestamp matching. This parser only performs light wrapper cleanup;
    semantic/content validation still decides whether the result is acceptable.
    """
    cleaned = result_text.strip()
    cleaned = re.sub(r"^```(?:\w+)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    output_match = re.search(
        r"<output>\s*(.*?)\s*</output>", cleaned, flags=re.IGNORECASE | re.DOTALL
    )
    if output_match:
        cleaned = output_match.group(1).strip()

    cleaned = re.sub(
        r"^\s*(?:output|result|answer|corrected(?: output)?|分句结果|输出)\s*[:：]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*\n+\s*", " ", cleaned)
    cleaned = re.sub(r"\s*<br\s*/?>\s*", "<br>", cleaned, flags=re.IGNORECASE)

    return [segment.strip() for segment in cleaned.split("<br>") if segment.strip()]


def _validate_split_result(
    original_text: str,
    split_result: List[str],
    max_word_count_cjk: int,
    max_word_count_english: int,
    hard_max_word_count_english: int | None = None,
) -> Tuple[bool, str]:
    """验证断句结果: 内容一致性、Segments数量、长度限制

    Returns: (is_valid, error_feedback)
    """
    # 检查是否为空
    if not split_result:
        return False, "No segments found. Split the text with <br> tags."

    # 检查内容是否被修改（使用difflib精确定位差异）
    original_cleaned = re.sub(r"\s+", " ", original_text)
    text_is_cjk = is_mainly_cjk(original_cleaned)

    merged_char = "" if text_is_cjk else " "
    merged = merged_char.join(split_result)
    merged_cleaned = re.sub(r"\s+", " ", merged)

    # Splitting may adjust punctuation and whitespace, but it must never remove,
    # insert, or rewrite spoken words. Character similarity can hide a handful
    # of dropped filler words in a long batch, so lock the lexical sequence for
    # Latin-language input before applying the more permissive punctuation check.
    original_has_cjk = bool(
        re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", original_cleaned)
    )
    if not text_is_cjk and not original_has_cjk:
        original_tokens = re.findall(
            r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", original_cleaned.lower()
        )
        merged_tokens = re.findall(
            r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", merged_cleaned.lower()
        )
        if original_tokens != merged_tokens:
            token_matcher = difflib.SequenceMatcher(None, original_tokens, merged_tokens)
            token_differences = []
            for opcode, a0, a1, b0, b1 in token_matcher.get_opcodes():
                if opcode == "equal":
                    continue
                token_differences.append(
                    f"{opcode}: expected {original_tokens[a0:a1]!r}, "
                    f"got {merged_tokens[b0:b1]!r}"
                )
                if len(token_differences) >= 5:
                    break
            return (
                False,
                "Source words were modified. Only punctuation, whitespace, and <br> "
                "placement may change:\n- " + "\n- ".join(token_differences),
            )

    # 使用SequenceMatcher计算相似度和差异
    matcher = difflib.SequenceMatcher(None, original_cleaned, merged_cleaned)
    similarity_ratio = matcher.ratio()

    # 允许98%以上的相似度（容忍少量标点或空格差异）
    if similarity_ratio < 0.96:
        differences = []
        context_size = 5 if text_is_cjk else 20

        for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
            if opcode == "replace":
                # 获取前后文
                before = original_cleaned[max(0, a0 - context_size) : a0]
                orig_part = original_cleaned[a0:a1]
                after = original_cleaned[a1 : a1 + context_size]

                new_part = merged_cleaned[b0:b1]

                if orig_part.isspace() or new_part.isspace():
                    continue

                differences.append(
                    f"...{before}[{orig_part}]{after}... → changed to [{new_part}]"
                )

            elif opcode == "delete":
                before = original_cleaned[max(0, a0 - context_size) : a0]
                deleted_part = original_cleaned[a0:a1]
                after = original_cleaned[a1 : a1 + context_size]

                if deleted_part.isspace():
                    continue

                differences.append(f"...{before}[{deleted_part}]{after}... → deleted")

            elif opcode == "insert":
                # 对于插入，显示插入位置的上下文
                before = merged_cleaned[max(0, b0 - context_size) : b0]
                inserted_part = merged_cleaned[b0:b1]
                after = merged_cleaned[b1 : b1 + context_size]

                if inserted_part.isspace():
                    continue

                differences.append(
                    f"Wrongly inserted [{inserted_part}] between '...{before}' and '{after}...'"
                )

        if differences:
            error_msg = f"Content modified (similarity: {similarity_ratio:.1%}):\n"
            error_msg += "\n".join(f"- {diff}" for diff in differences)
            error_msg += (
                "\nKeep original text unchanged, only insert <br> between words."
            )
            return False, error_msg

    # 检查每段长度是否超限
    violations = []
    for i, segment in enumerate(split_result, 1):
        word_count = count_words(segment)

        max_allowed = (
            max_word_count_cjk
            if text_is_cjk
            else hard_max_word_count_english or max_word_count_english
        )

        if word_count > max_allowed:
            segment_preview = segment[:40] + "..." if len(segment) > 40 else segment
            violations.append(
                f"Segment {i} '{segment_preview}': {word_count} {'chars' if text_is_cjk else 'words'} > {max_allowed} limit"
            )

    if violations:
        error_msg = "Length violations:\n" + "\n".join(f"- {v}" for v in violations)
        error_msg += "\n\nSplit these long segments further with <br>, then output the COMPLETE text with ALL segments (not just the fixed ones)."
        return False, error_msg

    if not text_is_cjk and not original_has_cjk:
        dangling = [
            f"Segment {index} ends with an incomplete phrase: '{segment}'"
            for index, segment in enumerate(split_result[:-1], 1)
            if _has_dangling_english_tail(segment, split_result[index])
        ]
        if dangling:
            return (
                False,
                "Unnatural split boundaries:\n"
                + "\n".join(f"- {item}" for item in dangling)
                + "\nMove each <br> to an earlier or later natural clause boundary. "
                "Do not end a subtitle with a preposition, determiner, conjunction, "
                "hedge such as 'probably', or an incomplete phrase such as 'a lot'.",
            )

    return True, ""


if __name__ == "__main__":
    sample_text = "大家好我叫杨玉溪来自有着良好音乐氛围的福建厦门自记事起我眼中的世界就是朦胧的童话书是各色杂乱的线条电视机是颜色各异的雪花小伙伴是只听其声不便骑行的马赛克后来我才知道这是一种眼底黄斑疾病虽不至于失明但终身无法治愈"
    sentences = split_by_llm(sample_text)
    print(f"断句结果 ({len(sentences)} 段):")
    for i, seg in enumerate(sentences, 1):
        print(f"  {i}. {seg}")
