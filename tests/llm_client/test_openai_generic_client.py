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
    assert schema['required'] == ['items']
    assert schema['$defs']['NestedResponseModel']['additionalProperties'] is False
    assert schema['$defs']['NestedResponseModel']['required'] == ['name', 'values']


@pytest.mark.asyncio
async def test_generate_response_honors_per_call_max_tokens():
    # The client previously sent self.max_tokens (16384) for every call,
    # ignoring the per-call budget the caller passed.
    dummy_client = DummyClient()
    client = OpenAIGenericClient(
        config=LLMConfig(api_key='test-key', base_url='https://llm.test/v1'),
        client=dummy_client,
    )

    await client._generate_response(messages=[], response_model=ResponseModel, max_tokens=512)

    assert dummy_client.chat.completions.create_calls[0]['max_tokens'] == 512


@pytest.mark.asyncio
async def test_generate_response_routes_small_model_size_to_small_model():
    from graphiti_core.llm_client.config import ModelSize

    dummy_client = DummyClient()
    client = OpenAIGenericClient(
        config=LLMConfig(
            api_key='test-key',
            base_url='https://llm.test/v1',
            model='primary-model',
            small_model='small-model',
        ),
        client=dummy_client,
    )

    await client._generate_response(
        messages=[], response_model=ResponseModel, model_size=ModelSize.small
    )
    await client._generate_response(
        messages=[], response_model=ResponseModel, model_size=ModelSize.medium
    )

    calls = dummy_client.chat.completions.create_calls
    assert calls[0]['model'] == 'small-model'
    assert calls[1]['model'] == 'primary-model'


@pytest.mark.asyncio
async def test_generate_response_falls_back_to_primary_without_small_model():
    from graphiti_core.llm_client.config import ModelSize

    dummy_client = DummyClient()
    client = OpenAIGenericClient(
        config=LLMConfig(api_key='test-key', base_url='https://llm.test/v1', model='primary-model'),
        client=dummy_client,
    )

    await client._generate_response(
        messages=[], response_model=ResponseModel, model_size=ModelSize.small
    )

    assert dummy_client.chat.completions.create_calls[0]['model'] == 'primary-model'


@pytest.mark.asyncio
async def test_retry_keeps_exactly_one_error_context_turn():
    # The retry loop previously APPENDED a new error turn per attempt and
    # resent the whole growing conversation — the cost amplifier.
    from graphiti_core.prompts.models import Message

    class FlakyChatCompletions(DummyChatCompletions):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        async def create(self, **kwargs):
            self.create_calls.append(kwargs)
            self.attempts += 1
            if self.attempts <= 2:
                message = SimpleNamespace(content='not json')
            else:
                message = SimpleNamespace(content='{"items":[]}')
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    dummy_client = DummyClient()
    dummy_client.chat.completions = FlakyChatCompletions()
    client = OpenAIGenericClient(
        config=LLMConfig(api_key='test-key', base_url='https://llm.test/v1'),
        client=dummy_client,
    )

    messages = [Message(role='system', content='sys'), Message(role='user', content='extract')]
    result = await client.generate_response(messages, response_model=ResponseModel)

    assert result == {'items': []}
    assert dummy_client.chat.completions.attempts == 3
    error_turns = [
        m for m in messages if m.content.startswith('The previous response attempt was invalid.')
    ]
    assert len(error_turns) == 1


@pytest.mark.asyncio
async def test_usage_callback_receives_token_counts_and_never_fails_the_call():
    class UsageChatCompletions(DummyChatCompletions):
        async def create(self, **kwargs):
            self.create_calls.append(kwargs)
            message = SimpleNamespace(content='{"items":[]}')
            choice = SimpleNamespace(message=message)
            usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120)
            return SimpleNamespace(choices=[choice], usage=usage)

    dummy_client = DummyClient()
    dummy_client.chat.completions = UsageChatCompletions()
    client = OpenAIGenericClient(
        config=LLMConfig(api_key='test-key', base_url='https://llm.test/v1', model='m'),
        client=dummy_client,
    )
    received = []
    client.usage_callback = lambda *args: received.append(args)

    result = await client._generate_response(messages=[], response_model=ResponseModel)
    assert result == {'items': []}
    assert received == [('m', 100, 20, 120)]

    # A raising callback must never fail the call that produced the tokens.
    def boom(*_args):
        raise RuntimeError('telemetry boom')

    client.usage_callback = boom
    result = await client._generate_response(messages=[], response_model=ResponseModel)
    assert result == {'items': []}
