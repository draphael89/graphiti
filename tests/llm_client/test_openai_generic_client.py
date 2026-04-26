from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient


class NestedResponseModel(BaseModel):
    name: str
    values: list[str] = Field(default_factory=list)


class ResponseModel(BaseModel):
    items: list[NestedResponseModel]


class DummyChatCompletions:
    def __init__(self):
        self.create_calls = []

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        message = SimpleNamespace(content='{"items":[]}')
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class DummyChat:
    def __init__(self):
        self.completions = DummyChatCompletions()


class DummyClient:
    def __init__(self):
        self.chat = DummyChat()


@pytest.mark.asyncio
async def test_structured_completion_adds_additional_properties_false_to_objects():
    dummy_client = DummyClient()
    client = OpenAIGenericClient(
        config=LLMConfig(api_key='test-key', base_url='https://llm.test/v1'),
        client=dummy_client,
    )

    await client._generate_response(messages=[], response_model=ResponseModel)

    response_format = dummy_client.chat.completions.create_calls[0]['response_format']
    schema = response_format['json_schema']['schema']

    assert response_format['type'] == 'json_schema'
    assert schema['additionalProperties'] is False
    assert schema['$defs']['NestedResponseModel']['additionalProperties'] is False
