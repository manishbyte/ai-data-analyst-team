
from unittest.mock import patch

import pytest

from app.config import Settings
from llm.client import GroqClient, LLMConfigurationError


def test_missing_api_key_is_rejected():
    settings = Settings(
        groq_api_key=None,
        groq_model="test-model",
    )

    with patch("llm.client.get_settings", return_value=settings):
        with pytest.raises(
            LLMConfigurationError,
            match="GROQ_API_KEY",
        ):
            GroqClient()


def test_missing_model_is_rejected():
    settings = Settings(
        groq_api_key="test-key",
        groq_model=None,
    )

    with patch("llm.client.get_settings", return_value=settings):
        with pytest.raises(
            LLMConfigurationError,
            match="GROQ_MODEL",
        ):
            GroqClient()