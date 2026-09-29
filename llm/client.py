
from groq import AsyncGroq

from app.config import get_settings


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM configuration is missing."""


class GroqClient:
    def __init__(self) -> None:
        settings = get_settings()

        if not settings.groq_api_key:
            raise LLMConfigurationError(
                "GROQ_API_KEY is not configured."
            )

        if not settings.groq_model:
            raise LLMConfigurationError(
                "GROQ_MODEL is not configured."
            )

        self.model = settings.groq_model
        self.client = AsyncGroq(
            api_key=settings.groq_api_key.get_secret_value(),
            timeout=30.0,
            max_retries=2,
        )

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError("Groq returned an empty response.")

        return content.strip()

    async def close(self) -> None:
        await self.client.close()