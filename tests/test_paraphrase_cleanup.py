from types import SimpleNamespace

import pytest
from google.genai import errors

from config import Config
from paraphrase import ParaphraseError, clean_output, paraphrase, system_prompt

KEY = "secret-key-abc"
CFG = Config(api_key=KEY)


# ---------------------------------------------------------------------------
# clean_output
# ---------------------------------------------------------------------------

def test_strip_whitespace():
    assert clean_output("  hello world \n") == "hello world"


@pytest.mark.parametrize("raw", [
    "```\nWe will sync tomorrow.\n```",
    "```text\nWe will sync tomorrow.\n```",
    "  ```markdown\nWe will sync tomorrow.```  ",
])
def test_code_fence_removed(raw):
    assert clean_output(raw) == "We will sync tomorrow."


def test_code_fence_multiline_kept_inside():
    assert clean_output("```\nline 1\nline 2\n```") == "line 1\nline 2"


def test_inline_fence_not_removed():
    assert clean_output("Run `pip install` first.") == "Run `pip install` first."


def test_code_fence_kept_when_original_is_fenced():
    raw = "```\nprint(1)\n```"
    assert clean_output(raw, original="```\nprint( 1 )\n```") == raw


@pytest.mark.parametrize("raw,expected", [
    ('"We will sync tomorrow."', "We will sync tomorrow."),
    ("“我们明天同步一下。”", "我们明天同步一下。"),
    ("「我们明天同步一下。」", "我们明天同步一下。"),
])
def test_outer_quotes_removed(raw, expected):
    assert clean_output(raw, original="嗯 那个 我们明天 sync 一下") == expected


@pytest.mark.parametrize("original", ['"quoted" he said', "“引用”开头", "「引用」开头"])
def test_outer_quotes_kept_when_original_starts_with_quote(original):
    assert clean_output('"Done."', original=original) == '"Done."'


def test_quotes_not_wrapping_whole_text_kept():
    raw = '"A" and "B"'
    assert clean_output(raw) == raw


def test_unpaired_quote_kept():
    assert clean_output('"Hello') == '"Hello'


@pytest.mark.parametrize("raw", [
    "改写后：我们明天同步。",
    "改写后:我们明天同步。",
    "改写结果： 我们明天同步。",
    "Rewritten: 我们明天同步。",
    "rewritten text: 我们明天同步。",
    "Here is the rewritten text: 我们明天同步。",
])
def test_prefix_removed(raw):
    assert clean_output(raw) == "我们明天同步。"


def test_prefix_then_quotes_removed():
    assert clean_output('Rewritten: "We will sync."') == "We will sync."


def test_prefix_kept_when_original_has_it():
    assert clean_output("Result: we shipped.", original="Result: um we like shipped") == "Result: we shipped."


def test_prefix_word_without_colon_kept():
    assert clean_output("Rewritten docs are ready.") == "Rewritten docs are ready."


@pytest.mark.parametrize("raw", [None, "", "   ", "```\n```", '""', "“”", "改写后：", "Rewritten: \"\""])
def test_empty_result_raises(raw):
    with pytest.raises(ParaphraseError) as exc:
        clean_output(raw)
    assert exc.value.kind == "empty"


# ---------------------------------------------------------------------------
# paraphrase() 用假 client，不调用真实 Gemini
# ---------------------------------------------------------------------------

class FakeModels:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        r = self.results.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


def fake_client(*results):
    return SimpleNamespace(models=FakeModels(results))


def resp(text, block_reason=None, finish_reason="STOP"):
    return SimpleNamespace(
        text=text,
        prompt_feedback=SimpleNamespace(block_reason=block_reason) if block_reason else None,
        candidates=[SimpleNamespace(finish_reason=finish_reason)],
    )


def api_error(code, message="boom", status="X"):
    return errors.APIError(code, {"error": {"code": code, "message": message, "status": status}})


def test_paraphrase_success_and_request_shape():
    client = fake_client(resp('"We will sync tomorrow."'))
    assert paraphrase("um we like sync tomorrow", cfg=CFG, client=client) == "We will sync tomorrow."
    call = client.models.calls[0]
    assert call["model"] == CFG.model
    assert call["contents"] == "um we like sync tomorrow"
    assert "text-rewriting engine" in call["config"].system_instruction
    assert call["config"].max_output_tokens == 2048
    assert call["config"].temperature is None


def test_rate_limit_retries_once_then_succeeds():
    sleeps = []
    client = fake_client(api_error(429), resp("ok"))
    assert paraphrase("x", cfg=CFG, client=client, sleep=sleeps.append) == "ok"
    assert sleeps == [1.0]
    assert len(client.models.calls) == 2


def test_rate_limit_twice_fails():
    sleeps = []
    client = fake_client(api_error(429), api_error(429), resp("never"))
    with pytest.raises(ParaphraseError) as exc:
        paraphrase("x", cfg=CFG, client=client, sleep=sleeps.append)
    assert exc.value.kind == "rate_limit"
    assert sleeps == [1.0]
    assert len(client.models.calls) == 2


@pytest.mark.parametrize("err", [
    api_error(401),
    api_error(403),
    api_error(400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT API_KEY_INVALID"),
])
def test_auth_error_not_retried(err):
    client = fake_client(err, resp("never"))
    with pytest.raises(ParaphraseError) as exc:
        paraphrase("x", cfg=CFG, client=client, sleep=lambda s: None)
    assert exc.value.kind == "auth"
    assert ".env" in str(exc.value)
    assert len(client.models.calls) == 1


def test_network_error():
    client = fake_client(TimeoutError("timed out"))
    with pytest.raises(ParaphraseError) as exc:
        paraphrase("x", cfg=CFG, client=client)
    assert exc.value.kind == "network"


def test_api_error_message_redacts_key_and_text():
    text = "my private message"
    client = fake_client(api_error(500, f"bad {KEY} for {text}"))
    with pytest.raises(ParaphraseError) as exc:
        paraphrase(text, cfg=CFG, client=client)
    assert exc.value.kind == "api"
    assert KEY not in str(exc.value)
    assert text not in str(exc.value)


def test_blocked_response():
    client = fake_client(resp(None, block_reason="SAFETY"))
    with pytest.raises(ParaphraseError) as exc:
        paraphrase("x", cfg=CFG, client=client)
    assert exc.value.kind == "blocked"


def test_empty_response():
    client = fake_client(resp(""))
    with pytest.raises(ParaphraseError) as exc:
        paraphrase("x", cfg=CFG, client=client)
    assert exc.value.kind == "empty"


def test_lang_selects_prompt():
    client = fake_client(resp("a"), resp("b"))
    paraphrase("x", "zh", cfg=CFG, client=client)
    paraphrase("x", "en", cfg=CFG, client=client)
    zh, en = (c["config"].system_instruction for c in client.models.calls)
    assert zh == system_prompt("zh") and "Simplified Chinese" in zh
    assert en == system_prompt("en") and "in English" in en
    assert "{" not in zh and "{" not in en


def test_default_lang_is_chinese():
    client = fake_client(resp("a"))
    paraphrase("x", cfg=CFG, client=client)
    assert client.models.calls[0]["config"].system_instruction == system_prompt("zh")


def test_unknown_lang_rejected():
    with pytest.raises(ValueError):
        paraphrase("x", "fr", cfg=CFG, client=fake_client())


# ---------------------------------------------------------------------------
# check_preserved：@提及 / URL 不能被改
# ---------------------------------------------------------------------------

from paraphrase import check_preserved  # noqa: E402


@pytest.mark.parametrize("original,result", [
    ("@李四 那个 PR 我看了", "@李四 I reviewed the PR."),
    ("@李四那个 PR 我看了", "@李四 我看了 PR。"),          # 提及后紧跟正文
    ("ping @bob.smith please", "Please ping @bob.smith."),
    ("见 https://example.com/docs?id=42 里面", "See https://example.com/docs?id=42."),
    ("文档在https://example.com/a里", "文档见 https://example.com/a。"),  # URL 后直接接中文
    ("see https://example.com/x.", "See https://example.com/x"),  # 句末标点不算 URL
    ("mail a@b.com about it", "Email a@b.com about it."),   # 邮箱不算提及
    ("no mentions here", "No mentions."),
])
def test_preserved_ok(original, result):
    check_preserved(original, result)


@pytest.mark.parametrize("original,result", [
    ("@李四 那个 PR 我看了", "@Li Si I reviewed the PR."),
    ("@李四 那个 PR 我看了", "I reviewed the PR."),
    ("见 https://example.com/docs?id=42", "See https://example.com/docs."),
    ("见 https://example.com/docs", "See the docs."),
])
def test_preserved_violation(original, result):
    with pytest.raises(ParaphraseError) as exc:
        check_preserved(original, result)
    assert exc.value.kind == "altered"
    assert original not in str(exc.value)


def test_paraphrase_rejects_translated_mention():
    client = fake_client(resp("@Li Si I reviewed PR #482."))
    with pytest.raises(ParaphraseError) as exc:
        paraphrase("@李四 PR #482 我看了", "en", cfg=CFG, client=client)
    assert exc.value.kind == "altered"


def test_prompt_mentions_ste_and_language_reminder_last():
    for lang, tail in (("zh", "Simplified Chinese"), ("en", "English")):
        p = system_prompt(lang)
        assert "ASD-STE100" in p
        assert tail in p.strip().splitlines()[-1]
