import json
import sys
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PROVIDERS = {
    "GLM_API_KEY": ("https://open.bigmodel.cn/api/paas/v4/chat/completions", "glm-4.7-flash"),
    "DEEPSEEK_API_KEY": ("https://api.deepseek.com/chat/completions", "deepseek-flash"),
}


def load_config(path: Path) -> tuple[str, str, str]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError("请在 chat.py 同目录准备可读的 UTF-8 格式 .env 文件。") from error

    values = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if separator:
            values[name.strip()] = value.strip()

    configured_keys = [name for name in PROVIDERS if values.get(name)]
    if len(configured_keys) != 1:
        raise ValueError("请在 .env 中填写 GLM_API_KEY 或 DEEPSEEK_API_KEY，并将另一项留空。")

    key_name = configured_keys[0]
    url, model = PROVIDERS[key_name]
    return url, model, values[key_name]


def call_model(url: str, model: str, api_key: str, messages: list[dict[str, str]]) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "thinking": {"type": "disabled"},
        "max_tokens": 1024,
    }
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            body = response.read()
    except HTTPError as error:
        error.close()
        raise RuntimeError(f"API 返回 HTTP {error.code}") from error
    except (OSError, HTTPException, ValueError) as error:
        raise RuntimeError("请求无法发送或响应读取失败") from error

    try:
        choice = json.loads(body)["choices"][0]
        reply = choice["message"]["content"]
        finish_reason = choice["finish_reason"]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise RuntimeError("API 响应格式有误") from error

    if finish_reason == "length":
        raise RuntimeError("回复达到输出长度上限，请缩短问题或要求简短回答")
    if finish_reason != "stop" or not isinstance(reply, str) or not reply.strip():
        raise RuntimeError("API 未返回完整的文本回复")
    return reply


def main() -> int:
    try:
        url, model, api_key = load_config(Path(__file__).with_name(".env"))
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    history: list[dict[str, str]] = []
    print(f"当前模型为 {model}。输入 /exit 结束对话。")
    while True:
        question = input("你 > ").strip()
        if question == "/exit":
            return 0
        if not question:
            continue

        user_message = {"role": "user", "content": question}
        request_messages = history + [user_message]
        try:
            reply = call_model(url, model, api_key, request_messages)
        except RuntimeError as error:
            print(f"调用失败（{error}）。可以重新输入后再试。", file=sys.stderr)
            continue

        history.extend([user_message, {"role": "assistant", "content": reply}])
        print(f"助手 > {reply}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EOFError, KeyboardInterrupt):
        print()
