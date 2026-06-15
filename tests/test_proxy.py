from vllmstat.providers.proxy import (
    SSEAccumulator,
    aiohttp_available,
    endpoint_for,
    extract_prompt,
    parse_json_content,
    parse_proxy_addr,
)


def test_endpoint_for():
    assert endpoint_for("/v1/chat/completions") == "chat"
    assert endpoint_for("/v1/completions") == "completions"
    assert endpoint_for("/v1/embeddings") is None


def test_extract_prompt_chat_and_completions():
    assert (
        extract_prompt(
            "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi there"}]}
        )
        == "hi there"
    )
    multi = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {}},
                ],
            }
        ]
    }
    assert extract_prompt("/v1/chat/completions", multi) == "look"
    assert extract_prompt("/v1/completions", {"prompt": "once upon"}) == "once upon"


def test_sse_accumulator_chat_split_chunks():
    acc = SSEAccumulator("chat")
    acc.feed('data: {"choices":[{"delta":{"content":"Hel')
    acc.feed('lo"}}]}\n\ndata: {"choices":[{"delta":{"content":" there"}}]}\n\n')
    acc.feed("data: [DONE]\n\n")
    assert acc.text == "Hello there" and acc.done is True


def test_sse_accumulator_completions():
    acc = SSEAccumulator("completions")
    acc.feed('data: {"choices":[{"text":"abc"}]}\n\ndata: [DONE]\n\n')
    assert acc.text == "abc" and acc.done is True


def test_parse_json_content_chat_with_usage():
    text, pt, ct = parse_json_content(
        {
            "choices": [{"message": {"content": "hello there"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        },
        "chat",
    )
    assert text == "hello there" and pt == 5 and ct == 2


def test_parse_proxy_addr():
    assert parse_proxy_addr("9000") == ("0.0.0.0", 9000)
    assert parse_proxy_addr("127.0.0.1:9000") == ("127.0.0.1", 9000)


def test_aiohttp_available_is_bool():
    assert isinstance(aiohttp_available(), bool)
