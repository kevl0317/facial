"""AI model providers the judge can use.

Claude goes through the Anthropic SDK. Every other provider speaks the OpenAI-style
chat-completions API, so one client covers them all; only the base URL, the default
model and how strictly they can be held to the JSON schema differ.

    key:  "required" | "optional" | "none"   (API key)
    json: "schema" (strict JSON-schema output) | "object" (JSON mode) | "none"
"""

from __future__ import annotations

import os

PROVIDERS: dict[str, dict] = {
    "claude": {
        "name": "Claude", "maker": "Anthropic", "kind": "anthropic",
        "base_url": "https://api.anthropic.com", "model": "claude-opus-5-5",
        "models": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-haiku-4-5"],
        "env": ["ANTHROPIC_API_KEY"], "key": "required", "json": "schema",
        "key_url": "https://console.anthropic.com/settings/keys",
    },
    "openai": {
        "name": "GPT", "maker": "OpenAI", "kind": "chat",
        "base_url": "https://api.openai.com/v1", "model": "gpt-5-mini",
        "models": ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini"],
        "env": ["OPENAI_API_KEY"], "key": "required", "json": "schema",
        "key_url": "https://platform.openai.com/api-keys",
    },
    "deepseek": {
        "name": "DeepSeek", "maker": "DeepSeek", "kind": "chat",
        "base_url": "https://api.deepseek.com", "model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "env": ["DEEPSEEK_API_KEY"], "key": "required", "json": "object",
        "key_url": "https://platform.deepseek.com/api_keys",
    },
    "gemini": {
        "name": "Gemini", "maker": "Google", "kind": "chat",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-2.5-flash",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"],
        "env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"], "key": "required", "json": "object",
        "key_url": "https://aistudio.google.com/apikey",
    },
    "grok": {
        "name": "Grok", "maker": "xAI", "kind": "chat",
        "base_url": "https://api.x.ai/v1", "model": "grok-3-mini",
        "models": ["grok-4", "grok-3-mini", "grok-3"],
        "env": ["XAI_API_KEY"], "key": "required", "json": "schema",
        "key_url": "https://console.x.ai",
    },
    "mistral": {
        "name": "Mistral", "maker": "Mistral AI", "kind": "chat",
        "base_url": "https://api.mistral.ai/v1", "model": "mistral-small-latest",
        "models": ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest",
                   "magistral-medium-latest"],
        "env": ["MISTRAL_API_KEY"], "key": "required", "json": "object",
        "key_url": "https://console.mistral.ai/api-keys",
    },
    "qwen": {
        "name": "Qwen", "maker": "Alibaba", "kind": "chat",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus",
        "models": ["qwen-plus", "qwen-max", "qwen-turbo", "qwen-flash"],
        "env": ["DASHSCOPE_API_KEY"], "key": "required", "json": "object", "edit_base": True,
        "key_url": "https://modelstudio.console.alibabacloud.com/",
    },
    "kimi": {
        "name": "Kimi", "maker": "Moonshot", "kind": "chat",
        "base_url": "https://api.moonshot.ai/v1", "model": "kimi-k2-0905-preview",
        "models": ["kimi-k2-0905-preview", "kimi-k2-turbo-preview", "kimi-latest", "moonshot-v1-32k"],
        "env": ["MOONSHOT_API_KEY"], "key": "required", "json": "object", "edit_base": True,
        "key_url": "https://platform.moonshot.ai/console/api-keys",
    },
    "glm": {
        "name": "GLM", "maker": "Zhipu", "kind": "chat",
        "base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4.5-air",
        "models": ["glm-4.6", "glm-4.5", "glm-4.5-air", "glm-4-flash"],
        "env": ["ZHIPUAI_API_KEY", "ZAI_API_KEY"], "key": "required", "json": "object", "edit_base": True,
        "key_url": "https://open.bigmodel.cn/usercenter/apikeys",
    },
    "groq": {
        "name": "Groq", "maker": "Groq", "kind": "chat",
        "base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile",
        "models": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "moonshotai/kimi-k2-instruct"],
        "env": ["GROQ_API_KEY"], "key": "required", "json": "object",
        "key_url": "https://console.groq.com/keys",
    },
    "openrouter": {
        "name": "OpenRouter", "maker": "OpenRouter", "kind": "chat",
        "base_url": "https://openrouter.ai/api/v1", "model": "google/gemini-2.5-flash",
        "models": ["google/gemini-2.5-flash", "openai/gpt-5-mini", "deepseek/deepseek-chat",
                   "anthropic/claude-sonnet-4.5", "meta-llama/llama-3.3-70b-instruct"],
        "env": ["OPENROUTER_API_KEY"], "key": "required", "json": "object",
        "key_url": "https://openrouter.ai/keys",
    },
    "ollama": {
        "name": "Ollama", "maker": "on this computer", "kind": "chat",
        "base_url": "http://localhost:11434/v1", "model": "llama3.2",
        "models": ["llama3.2", "qwen3", "gemma3", "deepseek-r1"],
        "env": [], "key": "none", "json": "object", "edit_base": True,
        "key_url": "https://ollama.com/download",
    },
    "custom": {
        "name": "AI", "maker": "other OpenAI-compatible", "kind": "chat",
        "base_url": "", "model": "", "models": [],
        "env": ["OPENAI_COMPATIBLE_API_KEY"], "key": "optional", "json": "object", "edit_base": True,
        "key_url": "",
    },
}

DEFAULT_PROVIDER = "claude"
RULES = "heuristic"

# Models that think before answering: they get a larger token budget.
THINKING_MODELS = (r"(^|/)(o\d|gpt-5)|reason|(^|[-/])r1\b|think|gemini-(2\.5|3)|grok-(3-mini|4)|qwq|magistral"
                   r"|glm-4\.[5-9]|gpt-oss")
# OpenAI models that take `reasoning_effort`.
OPENAI_REASONING = r"^(o\d|gpt-5)"


def provider(pid: str) -> dict:
    try:
        return {"id": pid, **PROVIDERS[pid]}
    except KeyError:
        raise ValueError(f"Unknown provider {pid!r}; choose one of: {', '.join(PROVIDERS)}") from None


def env_key(pid: str) -> str:
    """The provider's API key from the environment, if set."""
    return next((os.environ[e] for e in PROVIDERS[pid]["env"] if os.environ.get(e)), "")


def judge_tag(source: str, rules: str = "Rules") -> str:
    """Short label for a judgment's source, as shown on the overlay."""
    if source == "jev":
        return "Jev"
    return PROVIDERS[source]["name"] if source in PROVIDERS else rules


def judge_label(j: dict, rules: str = "Rules") -> str:
    """Who decided and who wrote a judgment, e.g. "Jev · Claude" (the footer credit)."""
    tag = judge_tag(j.get("source", ""), rules)
    return f"{tag} · {judge_tag(j['writer'])}" if j.get("writer") else tag
