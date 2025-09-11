import os
import pytest
import builtins

import importlib

import src.langchain_code.config_core as config_core


def test_normalize_gemini_env(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    importlib.reload(config_core) 

    assert os.environ["GOOGLE_API_KEY"] == "gem-key"
    assert os.environ["GEMINI_API_KEY"] == "gem-key"


@pytest.mark.parametrize("query,expected", [
    ("quick fix", "simple"),
    ("optimize database performance and add caching", "medium"),
    ("design microservices architecture with kubernetes orchestration", "complex"),
    ("implement enterprise-grade deep-learning with distributed infrastructure", "overly_complex"),
])
def test_classify_complexity(query, expected):
    r = config_core.IntelligentLLMRouter()
    assert r.classify_complexity(query) == expected


@pytest.mark.parametrize("cli,expected", [
    ("claude", "anthropic"),
    ("anthropic", "anthropic"),
    ("gemini", "gemini"),
    ("google", "gemini"),
    ("gpt", "openai"),
    ("openai", "openai"),
    ("ollama", "ollama"),
])
def test_resolve_provider_cli(cli, expected):
    assert config_core.resolve_provider(cli) == expected


def test_resolve_provider_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert config_core.resolve_provider(None) == "openai"

    monkeypatch.setenv("LLM_PROVIDER", "foobar")
    assert config_core.resolve_provider(None) == "gemini"


def test_get_model_info_defaults():
    info = config_core.get_model_info("anthropic")
    assert info["langchain_model_name"] == "claude-3-7-sonnet-20250219"
    assert info["complexity"] == "default"

    info = config_core.get_model_info("gemini")
    assert info["langchain_model_name"] == "gemini-2.0-flash"

    info = config_core.get_model_info("openai")
    assert info["langchain_model_name"] == "gpt-4o-mini"

    info = config_core.get_model_info("ollama")
    assert info["provider"] == "ollama"
    assert "langchain_model_name" in info


def test_get_model_info_with_query():
    query = "design kubernetes microservices architecture"
    info = config_core.get_model_info("gemini", query)
    assert info["provider"] == "gemini"
    assert info["complexity"] in {"complex", "overly_complex"}
    assert "reasoning_strength" in info


def test_pick_default_ollama_model(monkeypatch):
    monkeypatch.setenv("LANGCODE_OLLAMA_MODEL", "custom-model")
    assert config_core._pick_default_ollama_model() == "custom-model"

    monkeypatch.delenv("LANGCODE_OLLAMA_MODEL", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "env-ollama")
    assert config_core._pick_default_ollama_model() == "env-ollama"

    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert config_core._pick_default_ollama_model() == "llama3.1"
