"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import numpy as np
import pytest

import graphiti_core.graphiti as graphiti_module
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import CommunityEdge, EntityEdge, EpisodicEdge
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters
from graphiti_core.search.search_utils import (
    community_fulltext_search,
    community_similarity_search,
    edge_bfs_search,
    edge_fulltext_search,
    edge_similarity_search,
    episode_fulltext_search,
    episode_mentions_reranker,
    get_communities_by_nodes,
    get_edge_invalidation_candidates,
    get_embeddings_for_communities,
    get_embeddings_for_edges,
    get_embeddings_for_nodes,
    get_mentioned_nodes,
    get_relevant_edges,
    get_relevant_nodes,
    node_bfs_search,
    node_distance_reranker,
    node_fulltext_search,
    node_similarity_search,
)
from graphiti_core.utils.bulk_utils import RawEpisode, add_nodes_and_edges_bulk
from graphiti_core.utils.maintenance.community_operations import (
    determine_entity_community,
    get_community_clusters,
    remove_communities,
)
from graphiti_core.utils.maintenance.edge_operations import filter_existing_duplicate_of_edges
from tests.helpers_test import (
    GraphProvider,
    assert_entity_edge_equals,
    assert_entity_node_equals,
    assert_episodic_edge_equals,
    assert_episodic_node_equals,
    get_edge_count,
    get_node_count,
    group_id,
    group_id_2,
)

pytest_plugins = ('pytest_asyncio',)


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM"""
    mock_llm = Mock(spec=LLMClient)
    mock_llm.config = Mock()
    mock_llm.model = 'test-model'
    mock_llm.small_model = 'test-small-model'
    mock_llm.temperature = 0.0
    mock_llm.max_tokens = 1000
    mock_llm.cache_enabled = False
    mock_llm.cache_dir = None

    # Mock the public method that's actually called
    mock_llm.generate_response = Mock()
    mock_llm.generate_response.return_value = {
        'tool_calls': [
            {
                'name': 'extract_entities',
                'arguments': {'entities': [{'entity': 'test_entity', 'entity_type': 'test_type'}]},
            }
        ]
    }

    return mock_llm


@pytest.fixture
def mock_cross_encoder_client():
    """Create a mock LLM"""
    mock_llm = Mock(spec=CrossEncoderClient)
    mock_llm.config = Mock()

    # Mock the public method that's actually called
    mock_llm.rerank = Mock()
    mock_llm.rerank.return_value = {
        'tool_calls': [
            {
                'name': 'extract_entities',
                'arguments': {'entities': [{'entity': 'test_entity', 'entity_type': 'test_type'}]},
            }
        ]
    }

    return mock_llm


class NoIoGraphDriver(GraphDriver):
    provider = GraphProvider.NEO4J
    graph_operations_interface = None

    def __init__(self, database='default_db'):
        self._database = database

    async def execute_query(self, cypher_query_, **kwargs):
        raise AssertionError('execute_query should not be called')

    def session(self, database=None):
        raise AssertionError('session should not be called')

    async def close(self):
        pass

    async def delete_all_indexes(self):
        pass

    async def build_indices_and_constraints(self, delete_existing=False):
        pass

    def clone(self, database):
        return NoIoGraphDriver(database=database)


def no_io_graphiti(mock_llm_client, mock_embedder, mock_cross_encoder_client):
    return Graphiti(
        graph_driver=NoIoGraphDriver(),
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )


def deterministic_episode(
    uuid,
    reference_time,
    *,
    name='deterministic episode',
    content='user: Alice likes Bob',
    episode_group_id=group_id,
    graphiti_ingest_complete=True,
):
    return EpisodicNode(
        uuid=uuid,
        name=name,
        labels=[],
        source=EpisodeType.message,
        content=content,
        source_description='test',
        group_id=episode_group_id,
        created_at=datetime.now(),
        valid_at=reference_time,
        graphiti_ingest_complete=graphiti_ingest_complete,
    )


def deterministic_raw_episode(
    uuid,
    reference_time,
    *,
    name='deterministic episode',
    content='user: Alice likes Bob',
):
    return RawEpisode(
        uuid=uuid,
        name=name,
        content=content,
        source_description='test',
        source=EpisodeType.message,
        reference_time=reference_time,
    )


@pytest.mark.asyncio
async def test_add_episode_with_new_uuid_creates_episode(
    monkeypatch, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    requested_uuid = '11111111-1111-4111-8111-111111111111'

    async def missing_episode(cls, driver, uuid):
        raise NodeNotFoundError(uuid)

    async def no_previous_episodes(self, *args, **kwargs):
        return []

    async def no_extracted_nodes(*args, **kwargs):
        return [], {}

    async def no_resolved_nodes(*args, **kwargs):
        return [], {}, []

    async def no_resolved_edges(self, *args, **kwargs):
        return [], [], []

    async def no_hydrated_nodes(*args, **kwargs):
        return []

    async def return_episode(self, episode, *args, **kwargs):
        return [], episode

    async def mark_complete(self, episodes):
        for episode in episodes:
            episode.graphiti_ingest_complete = True

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', classmethod(missing_episode))
    monkeypatch.setattr(Graphiti, 'retrieve_episodes', no_previous_episodes)
    monkeypatch.setattr(graphiti_module, 'extract_nodes', no_extracted_nodes)
    monkeypatch.setattr(graphiti_module, 'resolve_extracted_nodes', no_resolved_nodes)
    monkeypatch.setattr(Graphiti, '_extract_and_resolve_edges', no_resolved_edges)
    monkeypatch.setattr(graphiti_module, 'extract_attributes_from_nodes', no_hydrated_nodes)
    monkeypatch.setattr(Graphiti, '_process_episode_data', return_episode)
    monkeypatch.setattr(Graphiti, '_mark_deterministic_episodes_complete', mark_complete)

    graphiti = no_io_graphiti(mock_llm_client, mock_embedder, mock_cross_encoder_client)

    result = await graphiti.add_episode(
        name='deterministic episode',
        episode_body='user: Alice likes Bob',
        source_description='test',
        reference_time=datetime.now(),
        source=EpisodeType.message,
        group_id=group_id,
        uuid=requested_uuid,
    )

    assert result.episode.uuid == requested_uuid
    assert result.episode.group_id == group_id
    assert result.episode.content == 'user: Alice likes Bob'
    assert result.episode.graphiti_ingest_complete is True


@pytest.mark.asyncio
async def test_add_episode_with_existing_uuid_returns_without_processing(
    monkeypatch, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    requested_uuid = '11111111-1111-4111-8111-111111111111'
    reference_time = datetime.now()
    existing_episode = deterministic_episode(requested_uuid, reference_time)

    async def found_episode(cls, driver, uuid):
        return existing_episode

    async def should_not_retrieve_previous_episodes(self, *args, **kwargs):
        raise AssertionError('existing uuid replay should not retrieve previous episodes')

    async def should_not_extract_nodes(*args, **kwargs):
        raise AssertionError('existing uuid replay should not extract nodes')

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', classmethod(found_episode))
    monkeypatch.setattr(Graphiti, 'retrieve_episodes', should_not_retrieve_previous_episodes)
    monkeypatch.setattr(graphiti_module, 'extract_nodes', should_not_extract_nodes)

    graphiti = no_io_graphiti(mock_llm_client, mock_embedder, mock_cross_encoder_client)

    result = await graphiti.add_episode(
        name='deterministic episode',
        episode_body='user: Alice likes Bob',
        source_description='test',
        reference_time=reference_time,
        source=EpisodeType.message,
        group_id=group_id,
        uuid=requested_uuid,
        update_communities=True,
        previous_episode_uuids=['22222222-2222-4222-8222-222222222222'],
    )

    assert result.episode == existing_episode
    assert result.episodic_edges == []
    assert result.nodes == []
    assert result.edges == []
    assert result.communities == []
    assert result.community_edges == []


@pytest.mark.asyncio
async def test_add_episode_with_existing_uuid_rejects_saga_replay(
    monkeypatch, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    requested_uuid = '11111111-1111-4111-8111-111111111111'
    reference_time = datetime.now()
    existing_episode = deterministic_episode(requested_uuid, reference_time)

    async def found_episode(cls, driver, uuid):
        return existing_episode

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', classmethod(found_episode))

    graphiti = no_io_graphiti(mock_llm_client, mock_embedder, mock_cross_encoder_client)

    with pytest.raises(ValueError, match='saga replay'):
        await graphiti.add_episode(
            name='deterministic episode',
            episode_body='user: Alice likes Bob',
            source_description='test',
            reference_time=reference_time,
            source=EpisodeType.message,
            group_id=group_id,
            uuid=requested_uuid,
            saga='retry saga',
        )


@pytest.mark.asyncio
async def test_add_episode_with_existing_uuid_rejects_payload_mismatch(
    monkeypatch, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    requested_uuid = '11111111-1111-4111-8111-111111111111'
    reference_time = datetime.now()
    existing_episode = deterministic_episode(requested_uuid, reference_time)

    async def found_episode(cls, driver, uuid):
        return existing_episode

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', classmethod(found_episode))

    graphiti = no_io_graphiti(mock_llm_client, mock_embedder, mock_cross_encoder_client)

    with pytest.raises(ValueError, match='different payload'):
        await graphiti.add_episode(
            name='deterministic episode',
            episode_body='user: Alice likes Carol',
            source_description='test',
            reference_time=reference_time,
            source=EpisodeType.message,
            group_id=group_id,
            uuid=requested_uuid,
        )


@pytest.mark.asyncio
async def test_add_episode_with_existing_uuid_rejects_group_mismatch(
    monkeypatch, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    requested_uuid = '11111111-1111-4111-8111-111111111111'
    reference_time = datetime.now()
    existing_episode = deterministic_episode(
        requested_uuid,
        reference_time,
        episode_group_id=group_id_2,
    )

    async def found_episode(cls, driver, uuid):
        return existing_episode

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', classmethod(found_episode))

    graphiti = no_io_graphiti(mock_llm_client, mock_embedder, mock_cross_encoder_client)

    with pytest.raises(ValueError, match='different group_id'):
        await graphiti.add_episode(
            name='deterministic episode',
            episode_body='user: Alice likes Bob',
            source_description='test',
            reference_time=reference_time,
            source=EpisodeType.message,
            group_id=group_id,
            uuid=requested_uuid,
        )


@pytest.mark.asyncio
async def test_add_episode_with_existing_uuid_accepts_equivalent_reference_time(
    monkeypatch, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    requested_uuid = '11111111-1111-4111-8111-111111111111'
    reference_time = datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc)
    existing_episode = deterministic_episode(
        requested_uuid,
        datetime(2026, 4, 24, 8, 0, 0, tzinfo=timezone(timedelta(hours=-4))),
    )

    async def found_episode(cls, driver, uuid):
        return existing_episode

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', classmethod(found_episode))

    graphiti = no_io_graphiti(mock_llm_client, mock_embedder, mock_cross_encoder_client)

    result = await graphiti.add_episode(
        name='deterministic episode',
        episode_body='user: Alice likes Bob',
        source_description='test',
        reference_time=reference_time,
        source=EpisodeType.message,
        group_id=group_id,
        uuid=requested_uuid,
    )

    assert result.episode == existing_episode


@pytest.mark.asyncio
async def test_add_episode_with_existing_incomplete_uuid_reprocesses(
    monkeypatch, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    requested_uuid = '11111111-1111-4111-8111-111111111111'
    reference_time = datetime.now()
    existing_episode = deterministic_episode(
        requested_uuid,
        reference_time,
        graphiti_ingest_complete=False,
    )
    processed_episodes: list[EpisodicNode] = []
    marked_episodes: list[EpisodicNode] = []

    async def found_episode(cls, driver, uuid):
        return existing_episode

    async def no_previous_episodes(self, *args, **kwargs):
        return []

    async def no_extracted_nodes(*args, **kwargs):
        return [], {}

    async def no_resolved_nodes(*args, **kwargs):
        return [], {}, []

    async def no_resolved_edges(self, *args, **kwargs):
        return [], [], []

    async def no_hydrated_nodes(*args, **kwargs):
        return []

    async def capture_process(self, episode, *args, **kwargs):
        processed_episodes.append(episode)
        return [], episode

    async def capture_complete(self, episodes):
        marked_episodes.extend(episodes)
        for episode in episodes:
            episode.graphiti_ingest_complete = True

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', classmethod(found_episode))
    monkeypatch.setattr(Graphiti, 'retrieve_episodes', no_previous_episodes)
    monkeypatch.setattr(graphiti_module, 'extract_nodes', no_extracted_nodes)
    monkeypatch.setattr(graphiti_module, 'resolve_extracted_nodes', no_resolved_nodes)
    monkeypatch.setattr(Graphiti, '_extract_and_resolve_edges', no_resolved_edges)
    monkeypatch.setattr(graphiti_module, 'extract_attributes_from_nodes', no_hydrated_nodes)
    monkeypatch.setattr(Graphiti, '_process_episode_data', capture_process)
    monkeypatch.setattr(Graphiti, '_mark_deterministic_episodes_complete', capture_complete)

    graphiti = no_io_graphiti(mock_llm_client, mock_embedder, mock_cross_encoder_client)

    result = await graphiti.add_episode(
        name='deterministic episode',
        episode_body='user: Alice likes Bob',
        source_description='test',
        reference_time=reference_time,
        source=EpisodeType.message,
        group_id=group_id,
        uuid=requested_uuid,
    )

    assert [episode.uuid for episode in processed_episodes] == [requested_uuid]
    assert [episode.uuid for episode in marked_episodes] == [requested_uuid]
    assert result.episode.graphiti_ingest_complete is True


@pytest.mark.asyncio
async def test_add_episode_with_uuid_rejects_raw_content_disabled_replay(
    monkeypatch, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    requested_uuid = '11111111-1111-4111-8111-111111111111'
    reference_time = datetime.now()
    existing_episode = deterministic_episode(requested_uuid, reference_time, content='')

    async def found_episode(cls, driver, uuid):
        return existing_episode

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', classmethod(found_episode))

    graphiti = no_io_graphiti(mock_llm_client, mock_embedder, mock_cross_encoder_client)
    graphiti.store_raw_episode_content = False

    with pytest.raises(ValueError, match='deterministic UUIDs require raw episode content storage'):
        await graphiti.add_episode(
            name='deterministic episode',
            episode_body='user: Alice likes Bob',
            source_description='test',
            reference_time=reference_time,
            source=EpisodeType.message,
            group_id=group_id,
            uuid=requested_uuid,
        )


@pytest.mark.asyncio
async def test_add_episode_bulk_with_existing_uuid_returns_without_processing(
    monkeypatch, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    requested_uuid = '11111111-1111-4111-8111-111111111111'
    reference_time = datetime.now()
    existing_episode = deterministic_episode(requested_uuid, reference_time)

    async def found_episode(cls, driver, uuid):
        return existing_episode

    async def should_not_add_nodes_and_edges_bulk(*args, **kwargs):
        raise AssertionError('existing uuid bulk replay should not save graph data')

    async def should_not_retrieve_previous_episodes_bulk(*args, **kwargs):
        raise AssertionError('existing uuid bulk replay should not retrieve context')

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', classmethod(found_episode))
    monkeypatch.setattr(graphiti_module, 'add_nodes_and_edges_bulk', should_not_add_nodes_and_edges_bulk)
    monkeypatch.setattr(
        graphiti_module,
        'retrieve_previous_episodes_bulk',
        should_not_retrieve_previous_episodes_bulk,
    )

    graphiti = no_io_graphiti(mock_llm_client, mock_embedder, mock_cross_encoder_client)

    result = await graphiti.add_episode_bulk(
        [
            deterministic_raw_episode(requested_uuid, reference_time)
        ],
        group_id=group_id,
    )

    assert result.episodes == [existing_episode]
    assert result.episodic_edges == []
    assert result.nodes == []
    assert result.edges == []
    assert result.communities == []
    assert result.community_edges == []


@pytest.mark.asyncio
async def test_add_episode_bulk_with_existing_incomplete_uuid_reprocesses(
    monkeypatch, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    requested_uuid = '11111111-1111-4111-8111-111111111111'
    reference_time = datetime.now()
    existing_episode = deterministic_episode(
        requested_uuid,
        reference_time,
        graphiti_ingest_complete=False,
    )
    saved_episodes: list[EpisodicNode] = []
    marked_episodes: list[EpisodicNode] = []

    async def found_episode(cls, driver, uuid):
        return existing_episode

    async def capture_add_nodes_and_edges_bulk(
        driver, episodic_nodes, episodic_edges, entity_nodes, entity_edges, embedder
    ):
        saved_episodes.extend(episodic_nodes)

    async def no_previous_episodes(driver, episodes):
        return [(episode, []) for episode in episodes]

    async def no_extracted_nodes(self, *args, **kwargs):
        return {}, {}, []

    async def no_edges(*args, **kwargs):
        return []

    async def no_resolved_nodes_and_edges(self, *args, **kwargs):
        return [], [], [], {}

    async def capture_complete(self, episodes):
        marked_episodes.extend(episodes)
        for episode in episodes:
            episode.graphiti_ingest_complete = True

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', classmethod(found_episode))
    monkeypatch.setattr(graphiti_module, 'add_nodes_and_edges_bulk', capture_add_nodes_and_edges_bulk)
    monkeypatch.setattr(graphiti_module, 'retrieve_previous_episodes_bulk', no_previous_episodes)
    monkeypatch.setattr(Graphiti, '_extract_and_dedupe_nodes_bulk', no_extracted_nodes)
    monkeypatch.setattr(graphiti_module, 'dedupe_edges_bulk', no_edges)
    monkeypatch.setattr(Graphiti, '_resolve_nodes_and_edges_bulk', no_resolved_nodes_and_edges)
    monkeypatch.setattr(Graphiti, '_mark_deterministic_episodes_complete', capture_complete)

    graphiti = no_io_graphiti(mock_llm_client, mock_embedder, mock_cross_encoder_client)

    result = await graphiti.add_episode_bulk(
        [deterministic_raw_episode(requested_uuid, reference_time)],
        group_id=group_id,
    )

    assert [episode.uuid for episode in saved_episodes] == [requested_uuid]
    assert [episode.uuid for episode in marked_episodes] == [requested_uuid]
    assert result.episodes == [existing_episode]
    assert result.episodes[0].graphiti_ingest_complete is True


@pytest.mark.asyncio
async def test_add_episode_bulk_with_uuid_rejects_raw_content_disabled_replay(
    monkeypatch, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    requested_uuid = '11111111-1111-4111-8111-111111111111'
    reference_time = datetime.now()
    existing_episode = deterministic_episode(requested_uuid, reference_time, content='')

    async def found_episode(cls, driver, uuid):
        return existing_episode

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', classmethod(found_episode))

    graphiti = no_io_graphiti(mock_llm_client, mock_embedder, mock_cross_encoder_client)
    graphiti.store_raw_episode_content = False

    with pytest.raises(ValueError, match='deterministic UUIDs require raw episode content storage'):
        await graphiti.add_episode_bulk(
            [deterministic_raw_episode(requested_uuid, reference_time)],
            group_id=group_id,
        )


@pytest.mark.asyncio
async def test_add_episode_bulk_with_existing_uuid_rejects_saga_replay(
    monkeypatch, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    requested_uuid = '11111111-1111-4111-8111-111111111111'
    reference_time = datetime.now()
    existing_episode = deterministic_episode(requested_uuid, reference_time)

    async def found_episode(cls, driver, uuid):
        return existing_episode

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', classmethod(found_episode))

    graphiti = no_io_graphiti(mock_llm_client, mock_embedder, mock_cross_encoder_client)

    with pytest.raises(ValueError, match='saga replay'):
        await graphiti.add_episode_bulk(
            [deterministic_raw_episode(requested_uuid, reference_time)],
            group_id=group_id,
            saga='retry saga',
        )


@pytest.mark.asyncio
async def test_add_episode_bulk_with_new_uuid_saves_once_after_extraction(
    monkeypatch, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    requested_uuid = '11111111-1111-4111-8111-111111111111'
    reference_time = datetime.now()
    saved_batches: list[list[EpisodicNode]] = []

    async def missing_episode(cls, driver, uuid):
        raise NodeNotFoundError(uuid)

    async def capture_add_nodes_and_edges_bulk(
        driver, episodic_nodes, episodic_edges, entity_nodes, entity_edges, embedder
    ):
        saved_batches.append(list(episodic_nodes))

    async def no_previous_episodes(driver, episodes):
        return [(episode, []) for episode in episodes]

    async def no_extracted_nodes(self, *args, **kwargs):
        return {}, {}, []

    async def no_edges(*args, **kwargs):
        return []

    async def no_resolved_nodes_and_edges(self, *args, **kwargs):
        return [], [], [], {}

    async def mark_complete(self, episodes):
        for episode in episodes:
            episode.graphiti_ingest_complete = True

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', classmethod(missing_episode))
    monkeypatch.setattr(graphiti_module, 'add_nodes_and_edges_bulk', capture_add_nodes_and_edges_bulk)
    monkeypatch.setattr(graphiti_module, 'retrieve_previous_episodes_bulk', no_previous_episodes)
    monkeypatch.setattr(Graphiti, '_extract_and_dedupe_nodes_bulk', no_extracted_nodes)
    monkeypatch.setattr(graphiti_module, 'dedupe_edges_bulk', no_edges)
    monkeypatch.setattr(Graphiti, '_resolve_nodes_and_edges_bulk', no_resolved_nodes_and_edges)
    monkeypatch.setattr(Graphiti, '_mark_deterministic_episodes_complete', mark_complete)

    graphiti = no_io_graphiti(mock_llm_client, mock_embedder, mock_cross_encoder_client)

    result = await graphiti.add_episode_bulk(
        [
            deterministic_raw_episode(requested_uuid, reference_time)
        ],
        group_id=group_id,
    )

    assert result.episodes[0].uuid == requested_uuid
    assert result.episodes[0].content == 'user: Alice likes Bob'
    assert result.episodes[0].graphiti_ingest_complete is True
    assert [[episode.uuid for episode in batch] for batch in saved_batches] == [[requested_uuid]]


@pytest.mark.asyncio
async def test_add_episode_bulk_with_mixed_existing_and_new_uuids_preserves_order(
    monkeypatch, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    existing_uuid = '11111111-1111-4111-8111-111111111111'
    new_uuid = '22222222-2222-4222-8222-222222222222'
    reference_time = datetime.now()
    existing_episode = deterministic_episode(
        existing_uuid,
        reference_time,
        name='existing deterministic episode',
    )
    saved_episodes: list[EpisodicNode] = []

    async def lookup_episode(cls, driver, uuid):
        if uuid == existing_uuid:
            return existing_episode
        raise NodeNotFoundError(uuid)

    async def capture_add_nodes_and_edges_bulk(
        driver, episodic_nodes, episodic_edges, entity_nodes, entity_edges, embedder
    ):
        saved_episodes.extend(episodic_nodes)

    async def no_previous_episodes(driver, episodes):
        return [(episode, []) for episode in episodes]

    async def no_extracted_nodes(self, *args, **kwargs):
        return {}, {}, []

    async def no_edges(*args, **kwargs):
        return []

    async def no_resolved_nodes_and_edges(self, *args, **kwargs):
        return [], [], [], {}

    async def mark_complete(self, episodes):
        for episode in episodes:
            episode.graphiti_ingest_complete = True

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', classmethod(lookup_episode))
    monkeypatch.setattr(graphiti_module, 'add_nodes_and_edges_bulk', capture_add_nodes_and_edges_bulk)
    monkeypatch.setattr(graphiti_module, 'retrieve_previous_episodes_bulk', no_previous_episodes)
    monkeypatch.setattr(Graphiti, '_extract_and_dedupe_nodes_bulk', no_extracted_nodes)
    monkeypatch.setattr(graphiti_module, 'dedupe_edges_bulk', no_edges)
    monkeypatch.setattr(Graphiti, '_resolve_nodes_and_edges_bulk', no_resolved_nodes_and_edges)
    monkeypatch.setattr(Graphiti, '_mark_deterministic_episodes_complete', mark_complete)

    graphiti = no_io_graphiti(mock_llm_client, mock_embedder, mock_cross_encoder_client)

    result = await graphiti.add_episode_bulk(
        [
            deterministic_raw_episode(
                existing_uuid,
                reference_time,
                name='existing deterministic episode',
            ),
            deterministic_raw_episode(
                new_uuid,
                reference_time,
                name='new deterministic episode',
            ),
        ],
        group_id=group_id,
    )

    assert [episode.uuid for episode in result.episodes] == [existing_uuid, new_uuid]
    assert [episode.uuid for episode in saved_episodes] == [new_uuid]


@pytest.mark.asyncio
async def test_add_episode_bulk_rejects_duplicate_new_uuid_in_same_request(
    monkeypatch, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    requested_uuid = '11111111-1111-4111-8111-111111111111'
    reference_time = datetime.now()

    async def missing_episode(cls, driver, uuid):
        raise NodeNotFoundError(uuid)

    async def should_not_retrieve_previous_episodes_bulk(*args, **kwargs):
        raise AssertionError('duplicate uuid bulk request should not retrieve context')

    async def should_not_add_nodes_and_edges_bulk(*args, **kwargs):
        raise AssertionError('duplicate uuid bulk request should not save graph data')

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', classmethod(missing_episode))
    monkeypatch.setattr(
        graphiti_module,
        'retrieve_previous_episodes_bulk',
        should_not_retrieve_previous_episodes_bulk,
    )
    monkeypatch.setattr(graphiti_module, 'add_nodes_and_edges_bulk', should_not_add_nodes_and_edges_bulk)

    duplicate_episodes = (
        deterministic_raw_episode(requested_uuid, reference_time),
        deterministic_raw_episode(
            requested_uuid,
            reference_time,
            content='user: Alice likes Carol',
        ),
    )

    for duplicate_episode in duplicate_episodes:
        graphiti = no_io_graphiti(mock_llm_client, mock_embedder, mock_cross_encoder_client)

        with pytest.raises(ValueError, match='appears more than once'):
            await graphiti.add_episode_bulk(
                [
                    deterministic_raw_episode(requested_uuid, reference_time),
                    duplicate_episode,
                ],
                group_id=group_id,
            )


@pytest.mark.asyncio
async def test_add_episode_bulk_with_existing_uuid_rejects_payload_mismatch(
    monkeypatch, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    requested_uuid = '11111111-1111-4111-8111-111111111111'
    reference_time = datetime.now()
    existing_episode = deterministic_episode(requested_uuid, reference_time)

    async def found_episode(cls, driver, uuid):
        return existing_episode

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', classmethod(found_episode))

    graphiti = no_io_graphiti(mock_llm_client, mock_embedder, mock_cross_encoder_client)

    with pytest.raises(ValueError, match='different payload'):
        await graphiti.add_episode_bulk(
            [
                deterministic_raw_episode(
                    requested_uuid,
                    reference_time,
                    content='user: Alice likes Carol',
                )
            ],
            group_id=group_id,
        )


@pytest.mark.asyncio
async def test_add_episode_bulk_with_existing_uuid_rejects_group_mismatch(
    monkeypatch, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    requested_uuid = '11111111-1111-4111-8111-111111111111'
    reference_time = datetime.now()
    existing_episode = deterministic_episode(
        requested_uuid,
        reference_time,
        episode_group_id=group_id_2,
    )

    async def found_episode(cls, driver, uuid):
        return existing_episode

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', classmethod(found_episode))

    graphiti = no_io_graphiti(mock_llm_client, mock_embedder, mock_cross_encoder_client)

    with pytest.raises(ValueError, match='different group_id'):
        await graphiti.add_episode_bulk(
            [deterministic_raw_episode(requested_uuid, reference_time)],
            group_id=group_id,
        )


@pytest.mark.asyncio
async def test_add_bulk(graph_driver, mock_llm_client, mock_embedder, mock_cross_encoder_client):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as test fails on FalkorDB')

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )

    await graphiti.build_indices_and_constraints()

    now = datetime.now()

    # Create episodic nodes
    episode_node_1 = EpisodicNode(
        name='test_episode',
        group_id=group_id,
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='conversation message',
        content='Alice likes Bob',
        valid_at=now,
        entity_edges=[],  # Filled in later
    )
    episode_node_2 = EpisodicNode(
        name='test_episode_2',
        group_id=group_id,
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='conversation message',
        content='Bob adores Alice',
        valid_at=now,
        entity_edges=[],  # Filled in later
    )

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        group_id=group_id,
        labels=['Entity', 'Person'],
        created_at=now,
        summary='test_entity_1 summary',
        attributes={'age': 30, 'location': 'New York'},
    )
    await entity_node_1.generate_name_embedding(mock_embedder)

    entity_node_2 = EntityNode(
        name='test_entity_2',
        group_id=group_id,
        labels=['Entity', 'Person2'],
        created_at=now,
        summary='test_entity_2 summary',
        attributes={'age': 25, 'location': 'Los Angeles'},
    )
    await entity_node_2.generate_name_embedding(mock_embedder)

    entity_node_3 = EntityNode(
        name='test_entity_3',
        group_id=group_id,
        labels=['Entity', 'City', 'Location'],
        created_at=now,
        summary='test_entity_3 summary',
        attributes={'age': 25, 'location': 'Los Angeles'},
    )
    await entity_node_3.generate_name_embedding(mock_embedder)

    entity_node_4 = EntityNode(
        name='test_entity_4',
        group_id=group_id,
        labels=['Entity'],
        created_at=now,
        summary='test_entity_4 summary',
        attributes={'age': 25, 'location': 'Los Angeles'},
    )
    await entity_node_4.generate_name_embedding(mock_embedder)

    # Create entity edges
    entity_edge_1 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        created_at=now,
        name='likes',
        fact='test_entity_1 relates to test_entity_2',
        episodes=[],
        expired_at=now,
        valid_at=now,
        invalid_at=now,
        group_id=group_id,
    )
    await entity_edge_1.generate_embedding(mock_embedder)

    entity_edge_2 = EntityEdge(
        source_node_uuid=entity_node_3.uuid,
        target_node_uuid=entity_node_4.uuid,
        created_at=now,
        name='relates_to',
        fact='test_entity_3 relates to test_entity_4',
        episodes=[],
        expired_at=now,
        valid_at=now,
        invalid_at=now,
        group_id=group_id,
    )
    await entity_edge_2.generate_embedding(mock_embedder)

    # Create episodic to entity edges
    episodic_edge_1 = EpisodicEdge(
        source_node_uuid=episode_node_1.uuid,
        target_node_uuid=entity_node_1.uuid,
        created_at=now,
        group_id=group_id,
    )
    episodic_edge_2 = EpisodicEdge(
        source_node_uuid=episode_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        created_at=now,
        group_id=group_id,
    )
    episodic_edge_3 = EpisodicEdge(
        source_node_uuid=episode_node_2.uuid,
        target_node_uuid=entity_node_3.uuid,
        created_at=now,
        group_id=group_id,
    )
    episodic_edge_4 = EpisodicEdge(
        source_node_uuid=episode_node_2.uuid,
        target_node_uuid=entity_node_4.uuid,
        created_at=now,
        group_id=group_id,
    )

    # Cross reference the ids
    episode_node_1.entity_edges = [entity_edge_1.uuid]
    episode_node_2.entity_edges = [entity_edge_2.uuid]
    entity_edge_1.episodes = [episode_node_1.uuid, episode_node_2.uuid]
    entity_edge_2.episodes = [episode_node_2.uuid]

    # Test add bulk
    await add_nodes_and_edges_bulk(
        graph_driver,
        [episode_node_1, episode_node_2],
        [episodic_edge_1, episodic_edge_2, episodic_edge_3, episodic_edge_4],
        [entity_node_1, entity_node_2, entity_node_3, entity_node_4],
        [entity_edge_1, entity_edge_2],
        mock_embedder,
    )

    node_ids = [
        episode_node_1.uuid,
        episode_node_2.uuid,
        entity_node_1.uuid,
        entity_node_2.uuid,
        entity_node_3.uuid,
        entity_node_4.uuid,
    ]
    edge_ids = [
        episodic_edge_1.uuid,
        episodic_edge_2.uuid,
        episodic_edge_3.uuid,
        episodic_edge_4.uuid,
        entity_edge_1.uuid,
        entity_edge_2.uuid,
    ]
    node_count = await get_node_count(graph_driver, node_ids)
    assert node_count == len(node_ids)
    edge_count = await get_edge_count(graph_driver, edge_ids)
    assert edge_count == len(edge_ids)

    # Test episodic nodes
    retrieved_episode = await EpisodicNode.get_by_uuid(graph_driver, episode_node_1.uuid)
    await assert_episodic_node_equals(retrieved_episode, episode_node_1)

    retrieved_episode = await EpisodicNode.get_by_uuid(graph_driver, episode_node_2.uuid)
    await assert_episodic_node_equals(retrieved_episode, episode_node_2)

    # Test entity nodes
    retrieved_entity_node = await EntityNode.get_by_uuid(graph_driver, entity_node_1.uuid)
    await assert_entity_node_equals(graph_driver, retrieved_entity_node, entity_node_1)

    retrieved_entity_node = await EntityNode.get_by_uuid(graph_driver, entity_node_2.uuid)
    await assert_entity_node_equals(graph_driver, retrieved_entity_node, entity_node_2)

    retrieved_entity_node = await EntityNode.get_by_uuid(graph_driver, entity_node_3.uuid)
    await assert_entity_node_equals(graph_driver, retrieved_entity_node, entity_node_3)

    retrieved_entity_node = await EntityNode.get_by_uuid(graph_driver, entity_node_4.uuid)
    await assert_entity_node_equals(graph_driver, retrieved_entity_node, entity_node_4)

    # Test episodic edges
    retrieved_episode_edge = await EpisodicEdge.get_by_uuid(graph_driver, episodic_edge_1.uuid)
    await assert_episodic_edge_equals(retrieved_episode_edge, episodic_edge_1)

    retrieved_episode_edge = await EpisodicEdge.get_by_uuid(graph_driver, episodic_edge_2.uuid)
    await assert_episodic_edge_equals(retrieved_episode_edge, episodic_edge_2)

    retrieved_episode_edge = await EpisodicEdge.get_by_uuid(graph_driver, episodic_edge_3.uuid)
    await assert_episodic_edge_equals(retrieved_episode_edge, episodic_edge_3)

    retrieved_episode_edge = await EpisodicEdge.get_by_uuid(graph_driver, episodic_edge_4.uuid)
    await assert_episodic_edge_equals(retrieved_episode_edge, episodic_edge_4)

    # Test entity edges
    retrieved_entity_edge = await EntityEdge.get_by_uuid(graph_driver, entity_edge_1.uuid)
    await assert_entity_edge_equals(graph_driver, retrieved_entity_edge, entity_edge_1)

    retrieved_entity_edge = await EntityEdge.get_by_uuid(graph_driver, entity_edge_2.uuid)
    await assert_entity_edge_equals(graph_driver, retrieved_entity_edge, entity_edge_2)


@pytest.mark.asyncio
async def test_remove_episode(
    graph_driver, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )

    await graphiti.build_indices_and_constraints()

    now = datetime.now()

    # Create episodic nodes
    episode_node = EpisodicNode(
        name='test_episode',
        group_id=group_id,
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='conversation message',
        content='Alice likes Bob',
        valid_at=now,
        entity_edges=[],  # Filled in later
    )

    # Create entity nodes
    alice_node = EntityNode(
        name='Alice',
        group_id=group_id,
        labels=['Entity', 'Person'],
        created_at=now,
        summary='Alice summary',
        attributes={'age': 30, 'location': 'New York'},
    )
    await alice_node.generate_name_embedding(mock_embedder)

    bob_node = EntityNode(
        name='Bob',
        group_id=group_id,
        labels=['Entity', 'Person2'],
        created_at=now,
        summary='Bob summary',
        attributes={'age': 25, 'location': 'Los Angeles'},
    )
    await bob_node.generate_name_embedding(mock_embedder)

    # Create entity to entity edge
    entity_edge = EntityEdge(
        source_node_uuid=alice_node.uuid,
        target_node_uuid=bob_node.uuid,
        created_at=now,
        name='likes',
        fact='Alice likes Bob',
        episodes=[],
        expired_at=now,
        valid_at=now,
        invalid_at=now,
        group_id=group_id,
    )
    await entity_edge.generate_embedding(mock_embedder)

    # Create episodic to entity edges
    episodic_alice_edge = EpisodicEdge(
        source_node_uuid=episode_node.uuid,
        target_node_uuid=alice_node.uuid,
        created_at=now,
        group_id=group_id,
    )
    episodic_bob_edge = EpisodicEdge(
        source_node_uuid=episode_node.uuid,
        target_node_uuid=bob_node.uuid,
        created_at=now,
        group_id=group_id,
    )

    # Cross reference the ids
    episode_node.entity_edges = [entity_edge.uuid]
    entity_edge.episodes = [episode_node.uuid]

    # Test add bulk
    await add_nodes_and_edges_bulk(
        graph_driver,
        [episode_node],
        [episodic_alice_edge, episodic_bob_edge],
        [alice_node, bob_node],
        [entity_edge],
        mock_embedder,
    )

    node_ids = [episode_node.uuid, alice_node.uuid, bob_node.uuid]
    edge_ids = [episodic_alice_edge.uuid, episodic_bob_edge.uuid, entity_edge.uuid]
    node_count = await get_node_count(graph_driver, node_ids)
    assert node_count == 3
    edge_count = await get_edge_count(graph_driver, edge_ids)
    assert edge_count == 3

    # Test remove episode
    await graphiti.remove_episode(episode_node.uuid)
    node_count = await get_node_count(graph_driver, node_ids)
    assert node_count == 0
    edge_count = await get_edge_count(graph_driver, edge_ids)
    assert edge_count == 0

    # Test add bulk again
    await add_nodes_and_edges_bulk(
        graph_driver,
        [episode_node],
        [episodic_alice_edge, episodic_bob_edge],
        [alice_node, bob_node],
        [entity_edge],
        mock_embedder,
    )
    node_count = await get_node_count(graph_driver, node_ids)
    assert node_count == 3
    edge_count = await get_edge_count(graph_driver, edge_ids)
    assert edge_count == 3


@pytest.mark.asyncio
async def test_graphiti_retrieve_episodes(
    graph_driver, mock_llm_client, mock_embedder, mock_cross_encoder_client
):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as test fails on FalkorDB')

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )

    await graphiti.build_indices_and_constraints()

    now = datetime.now()
    valid_at_1 = now - timedelta(days=2)
    valid_at_2 = now - timedelta(days=4)
    valid_at_3 = now - timedelta(days=6)

    # Create episodic nodes
    episode_node_1 = EpisodicNode(
        name='test_episode_1',
        labels=[],
        created_at=now,
        valid_at=valid_at_1,
        source=EpisodeType.message,
        source_description='conversation message',
        content='Test message 1',
        entity_edges=[],
        group_id=group_id,
    )
    episode_node_2 = EpisodicNode(
        name='test_episode_2',
        labels=[],
        created_at=now,
        valid_at=valid_at_2,
        source=EpisodeType.message,
        source_description='conversation message',
        content='Test message 2',
        entity_edges=[],
        group_id=group_id,
    )
    episode_node_3 = EpisodicNode(
        name='test_episode_3',
        labels=[],
        created_at=now,
        valid_at=valid_at_3,
        source=EpisodeType.message,
        source_description='conversation message',
        content='Test message 3',
        entity_edges=[],
        group_id=group_id,
    )

    # Save the nodes
    await episode_node_1.save(graph_driver)
    await episode_node_2.save(graph_driver)
    await episode_node_3.save(graph_driver)

    node_ids = [episode_node_1.uuid, episode_node_2.uuid, episode_node_3.uuid]
    node_count = await get_node_count(graph_driver, node_ids)
    assert node_count == 3

    # Retrieve episodes
    query_time = now - timedelta(days=3)
    episodes = await graphiti.retrieve_episodes(
        query_time, last_n=5, group_ids=[group_id], source=EpisodeType.message
    )
    assert len(episodes) == 2
    assert episodes[0].name == episode_node_3.name
    assert episodes[1].name == episode_node_2.name


@pytest.mark.asyncio
async def test_filter_existing_duplicate_of_edges(graph_driver, mock_embedder):
    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)
    entity_node_3 = EntityNode(
        name='test_entity_3',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_3.generate_name_embedding(mock_embedder)
    entity_node_4 = EntityNode(
        name='test_entity_4',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_4.generate_name_embedding(mock_embedder)

    # Save the nodes
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_node_3.save(graph_driver)
    await entity_node_4.save(graph_driver)

    node_ids = [entity_node_1.uuid, entity_node_2.uuid, entity_node_3.uuid, entity_node_4.uuid]
    node_count = await get_node_count(graph_driver, node_ids)
    assert node_count == 4

    # Create duplicate entity edge
    entity_edge = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        name='IS_DUPLICATE_OF',
        fact='test_entity_1 is a duplicate of test_entity_2',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_edge.generate_embedding(mock_embedder)
    await entity_edge.save(graph_driver)

    # Filter duplicate entity edges
    duplicate_node_tuples = [
        (entity_node_1, entity_node_2),
        (entity_node_3, entity_node_4),
    ]
    node_tuples = await filter_existing_duplicate_of_edges(graph_driver, duplicate_node_tuples)
    assert len(node_tuples) == 1
    assert [node.name for node in node_tuples[0]] == [entity_node_3.name, entity_node_4.name]


@pytest.mark.asyncio
async def test_determine_entity_community(graph_driver, mock_embedder):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as test fails on FalkorDB')

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)
    entity_node_3 = EntityNode(
        name='test_entity_3',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_3.generate_name_embedding(mock_embedder)
    entity_node_4 = EntityNode(
        name='test_entity_4',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_4.generate_name_embedding(mock_embedder)

    # Create entity edges
    entity_edge_1 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_4.uuid,
        name='RELATES_TO',
        fact='test_entity_1 relates to test_entity_4',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_edge_1.generate_embedding(mock_embedder)
    entity_edge_2 = EntityEdge(
        source_node_uuid=entity_node_2.uuid,
        target_node_uuid=entity_node_4.uuid,
        name='RELATES_TO',
        fact='test_entity_2 relates to test_entity_4',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_edge_2.generate_embedding(mock_embedder)
    entity_edge_3 = EntityEdge(
        source_node_uuid=entity_node_3.uuid,
        target_node_uuid=entity_node_4.uuid,
        name='RELATES_TO',
        fact='test_entity_3 relates to test_entity_4',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_edge_3.generate_embedding(mock_embedder)

    # Create community nodes
    community_node_1 = CommunityNode(
        name='test_community_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await community_node_1.generate_name_embedding(mock_embedder)
    community_node_2 = CommunityNode(
        name='test_community_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await community_node_2.generate_name_embedding(mock_embedder)

    # Create community to entity edges
    community_edge_1 = CommunityEdge(
        source_node_uuid=community_node_1.uuid,
        target_node_uuid=entity_node_1.uuid,
        created_at=datetime.now(),
        group_id=group_id,
    )
    community_edge_2 = CommunityEdge(
        source_node_uuid=community_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        created_at=datetime.now(),
        group_id=group_id,
    )
    community_edge_3 = CommunityEdge(
        source_node_uuid=community_node_2.uuid,
        target_node_uuid=entity_node_3.uuid,
        created_at=datetime.now(),
        group_id=group_id,
    )

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_node_3.save(graph_driver)
    await entity_node_4.save(graph_driver)
    await community_node_1.save(graph_driver)
    await community_node_2.save(graph_driver)

    await entity_edge_1.save(graph_driver)
    await entity_edge_2.save(graph_driver)
    await entity_edge_3.save(graph_driver)
    await community_edge_1.save(graph_driver)
    await community_edge_2.save(graph_driver)
    await community_edge_3.save(graph_driver)

    node_ids = [
        entity_node_1.uuid,
        entity_node_2.uuid,
        entity_node_3.uuid,
        entity_node_4.uuid,
        community_node_1.uuid,
        community_node_2.uuid,
    ]
    edge_ids = [
        entity_edge_1.uuid,
        entity_edge_2.uuid,
        entity_edge_3.uuid,
        community_edge_1.uuid,
        community_edge_2.uuid,
        community_edge_3.uuid,
    ]
    node_count = await get_node_count(graph_driver, node_ids)
    assert node_count == 6
    edge_count = await get_edge_count(graph_driver, edge_ids)
    assert edge_count == 6

    # Determine entity community
    community, is_new = await determine_entity_community(graph_driver, entity_node_4)
    assert community.name == community_node_1.name
    assert is_new

    # Add entity to community edge
    community_edge_4 = CommunityEdge(
        source_node_uuid=community_node_1.uuid,
        target_node_uuid=entity_node_4.uuid,
        created_at=datetime.now(),
        group_id=group_id,
    )
    await community_edge_4.save(graph_driver)

    # Determine entity community again
    community, is_new = await determine_entity_community(graph_driver, entity_node_4)
    assert community.name == community_node_1.name
    assert not is_new

    await remove_communities(graph_driver)
    node_count = await get_node_count(graph_driver, [community_node_1.uuid, community_node_2.uuid])
    assert node_count == 0


@pytest.mark.asyncio
async def test_get_community_clusters(graph_driver, mock_embedder):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as test fails on FalkorDB')

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)
    entity_node_3 = EntityNode(
        name='test_entity_3',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id_2,
    )
    await entity_node_3.generate_name_embedding(mock_embedder)
    entity_node_4 = EntityNode(
        name='test_entity_4',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id_2,
    )
    await entity_node_4.generate_name_embedding(mock_embedder)

    # Create entity edges
    entity_edge_1 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        name='RELATES_TO',
        fact='test_entity_1 relates to test_entity_2',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_edge_1.generate_embedding(mock_embedder)
    entity_edge_2 = EntityEdge(
        source_node_uuid=entity_node_3.uuid,
        target_node_uuid=entity_node_4.uuid,
        name='RELATES_TO',
        fact='test_entity_3 relates to test_entity_4',
        created_at=datetime.now(),
        group_id=group_id_2,
    )
    await entity_edge_2.generate_embedding(mock_embedder)

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_node_3.save(graph_driver)
    await entity_node_4.save(graph_driver)
    await entity_edge_1.save(graph_driver)
    await entity_edge_2.save(graph_driver)

    node_ids = [entity_node_1.uuid, entity_node_2.uuid, entity_node_3.uuid, entity_node_4.uuid]
    edge_ids = [entity_edge_1.uuid, entity_edge_2.uuid]
    node_count = await get_node_count(graph_driver, node_ids)
    assert node_count == 4
    edge_count = await get_edge_count(graph_driver, edge_ids)
    assert edge_count == 2

    # Get community clusters
    clusters = await get_community_clusters(graph_driver, group_ids=None)
    assert len(clusters) == 2
    assert len(clusters[0]) == 2
    assert len(clusters[1]) == 2
    entities_1 = set([node.name for node in clusters[0]])
    entities_2 = set([node.name for node in clusters[1]])
    assert entities_1 == set(['test_entity_1', 'test_entity_2']) or entities_2 == set(
        ['test_entity_1', 'test_entity_2']
    )
    assert entities_1 == set(['test_entity_3', 'test_entity_4']) or entities_2 == set(
        ['test_entity_3', 'test_entity_4']
    )


@pytest.mark.asyncio
async def test_get_mentioned_nodes(graph_driver, mock_embedder):
    # Create episodic nodes
    episodic_node_1 = EpisodicNode(
        name='test_episodic_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
        source=EpisodeType.message,
        source_description='test_source_description',
        content='test_content',
        valid_at=datetime.now(),
    )
    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)

    # Create episodic to entity edges
    episodic_edge_1 = EpisodicEdge(
        source_node_uuid=episodic_node_1.uuid,
        target_node_uuid=entity_node_1.uuid,
        created_at=datetime.now(),
        group_id=group_id,
    )

    # Save the graph
    await episodic_node_1.save(graph_driver)
    await entity_node_1.save(graph_driver)
    await episodic_edge_1.save(graph_driver)

    # Get mentioned nodes
    mentioned_nodes = await get_mentioned_nodes(graph_driver, [episodic_node_1])
    assert len(mentioned_nodes) == 1
    assert mentioned_nodes[0].name == entity_node_1.name


@pytest.mark.asyncio
async def test_get_communities_by_nodes(graph_driver, mock_embedder):
    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)

    # Create community nodes
    community_node_1 = CommunityNode(
        name='test_community_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await community_node_1.generate_name_embedding(mock_embedder)

    # Create community to entity edges
    community_edge_1 = CommunityEdge(
        source_node_uuid=community_node_1.uuid,
        target_node_uuid=entity_node_1.uuid,
        created_at=datetime.now(),
        group_id=group_id,
    )

    # Save the graph
    await entity_node_1.save(graph_driver)
    await community_node_1.save(graph_driver)
    await community_edge_1.save(graph_driver)

    # Get communities by nodes
    communities = await get_communities_by_nodes(graph_driver, [entity_node_1])
    assert len(communities) == 1
    assert communities[0].name == community_node_1.name


@pytest.mark.asyncio
async def test_edge_fulltext_search(
    graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder_client
):
    if graph_driver.provider == GraphProvider.KUZU:
        pytest.skip('Skipping as fulltext indexing not supported for Kuzu')

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )
    await graphiti.build_indices_and_constraints()

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)

    now = datetime.now()
    created_at = now
    expired_at = now + timedelta(days=6)
    valid_at = now + timedelta(days=2)
    invalid_at = now + timedelta(days=4)

    # Create entity edges
    entity_edge_1 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        name='RELATES_TO',
        fact='test_entity_1 relates to test_entity_2',
        created_at=created_at,
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
        group_id=group_id,
    )
    await entity_edge_1.generate_embedding(mock_embedder)

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_edge_1.save(graph_driver)

    # Search for entity edges
    search_filters = SearchFilters(
        node_labels=['Entity'],
        edge_types=['RELATES_TO'],
        created_at=[
            [DateFilter(date=created_at, comparison_operator=ComparisonOperator.equals)],
        ],
        expired_at=[
            [DateFilter(date=now, comparison_operator=ComparisonOperator.not_equals)],
        ],
        valid_at=[
            [
                DateFilter(
                    date=now + timedelta(days=1),
                    comparison_operator=ComparisonOperator.greater_than_equal,
                )
            ],
            [
                DateFilter(
                    date=now + timedelta(days=3),
                    comparison_operator=ComparisonOperator.less_than_equal,
                )
            ],
        ],
        invalid_at=[
            [
                DateFilter(
                    date=now + timedelta(days=3),
                    comparison_operator=ComparisonOperator.greater_than,
                )
            ],
            [
                DateFilter(
                    date=now + timedelta(days=5), comparison_operator=ComparisonOperator.less_than
                )
            ],
        ],
    )
    edges = await edge_fulltext_search(
        graph_driver, 'test_entity_1 relates to test_entity_2', search_filters, group_ids=[group_id]
    )
    assert len(edges) == 1
    assert edges[0].name == entity_edge_1.name


@pytest.mark.asyncio
async def test_edge_similarity_search(graph_driver, mock_embedder):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)

    now = datetime.now()
    created_at = now
    expired_at = now + timedelta(days=6)
    valid_at = now + timedelta(days=2)
    invalid_at = now + timedelta(days=4)

    # Create entity edges
    entity_edge_1 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        name='RELATES_TO',
        fact='test_entity_1 relates to test_entity_2',
        created_at=created_at,
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
        group_id=group_id,
    )
    await entity_edge_1.generate_embedding(mock_embedder)

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_edge_1.save(graph_driver)

    # Search for entity edges
    search_filters = SearchFilters(
        node_labels=['Entity'],
        edge_types=['RELATES_TO'],
        created_at=[
            [DateFilter(date=created_at, comparison_operator=ComparisonOperator.equals)],
        ],
        expired_at=[
            [DateFilter(date=now, comparison_operator=ComparisonOperator.not_equals)],
        ],
        valid_at=[
            [
                DateFilter(
                    date=now + timedelta(days=1),
                    comparison_operator=ComparisonOperator.greater_than_equal,
                )
            ],
            [
                DateFilter(
                    date=now + timedelta(days=3),
                    comparison_operator=ComparisonOperator.less_than_equal,
                )
            ],
        ],
        invalid_at=[
            [
                DateFilter(
                    date=now + timedelta(days=3),
                    comparison_operator=ComparisonOperator.greater_than,
                )
            ],
            [
                DateFilter(
                    date=now + timedelta(days=5), comparison_operator=ComparisonOperator.less_than
                )
            ],
        ],
    )
    edges = await edge_similarity_search(
        graph_driver,
        entity_edge_1.fact_embedding,
        entity_node_1.uuid,
        entity_node_2.uuid,
        search_filters,
        group_ids=[group_id],
    )
    assert len(edges) == 1
    assert edges[0].name == entity_edge_1.name


@pytest.mark.asyncio
async def test_edge_bfs_search(graph_driver, mock_embedder):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    # Create episodic nodes
    episodic_node_1 = EpisodicNode(
        name='test_episodic_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
        source=EpisodeType.message,
        source_description='test_source_description',
        content='test_content',
        valid_at=datetime.now(),
    )

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)
    entity_node_3 = EntityNode(
        name='test_entity_3',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_3.generate_name_embedding(mock_embedder)

    now = datetime.now()
    created_at = now
    expired_at = now + timedelta(days=6)
    valid_at = now + timedelta(days=2)
    invalid_at = now + timedelta(days=4)

    # Create entity edges
    entity_edge_1 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        name='RELATES_TO',
        fact='test_entity_1 relates to test_entity_2',
        created_at=created_at,
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
        group_id=group_id,
    )
    await entity_edge_1.generate_embedding(mock_embedder)
    entity_edge_2 = EntityEdge(
        source_node_uuid=entity_node_2.uuid,
        target_node_uuid=entity_node_3.uuid,
        name='RELATES_TO',
        fact='test_entity_2 relates to test_entity_3',
        created_at=created_at,
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
        group_id=group_id,
    )
    await entity_edge_2.generate_embedding(mock_embedder)

    # Create episodic to entity edges
    episodic_edge_1 = EpisodicEdge(
        source_node_uuid=episodic_node_1.uuid,
        target_node_uuid=entity_node_1.uuid,
        created_at=datetime.now(),
        group_id=group_id,
    )

    # Save the graph
    await episodic_node_1.save(graph_driver)
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_node_3.save(graph_driver)
    await entity_edge_1.save(graph_driver)
    await entity_edge_2.save(graph_driver)
    await episodic_edge_1.save(graph_driver)

    # Search for entity edges
    search_filters = SearchFilters(
        node_labels=['Entity'],
        edge_types=['RELATES_TO'],
        created_at=[
            [DateFilter(date=created_at, comparison_operator=ComparisonOperator.equals)],
        ],
        expired_at=[
            [DateFilter(date=now, comparison_operator=ComparisonOperator.not_equals)],
        ],
        valid_at=[
            [
                DateFilter(
                    date=now + timedelta(days=1),
                    comparison_operator=ComparisonOperator.greater_than_equal,
                )
            ],
            [
                DateFilter(
                    date=now + timedelta(days=3),
                    comparison_operator=ComparisonOperator.less_than_equal,
                )
            ],
        ],
        invalid_at=[
            [
                DateFilter(
                    date=now + timedelta(days=3),
                    comparison_operator=ComparisonOperator.greater_than,
                )
            ],
            [
                DateFilter(
                    date=now + timedelta(days=5), comparison_operator=ComparisonOperator.less_than
                )
            ],
        ],
    )

    # Test bfs from episodic node

    edges = await edge_bfs_search(
        graph_driver,
        [episodic_node_1.uuid],
        1,
        search_filters,
        group_ids=[group_id],
    )
    assert len(edges) == 0

    edges = await edge_bfs_search(
        graph_driver,
        [episodic_node_1.uuid],
        2,
        search_filters,
        group_ids=[group_id],
    )
    edges_deduplicated = set({edge.uuid: edge.fact for edge in edges}.values())
    assert len(edges_deduplicated) == 1
    assert edges_deduplicated == {'test_entity_1 relates to test_entity_2'}

    edges = await edge_bfs_search(
        graph_driver,
        [episodic_node_1.uuid],
        3,
        search_filters,
        group_ids=[group_id],
    )
    edges_deduplicated = set({edge.uuid: edge.fact for edge in edges}.values())
    assert len(edges_deduplicated) == 2
    assert edges_deduplicated == {
        'test_entity_1 relates to test_entity_2',
        'test_entity_2 relates to test_entity_3',
    }

    # Test bfs from entity node

    edges = await edge_bfs_search(
        graph_driver,
        [entity_node_1.uuid],
        1,
        search_filters,
        group_ids=[group_id],
    )
    edges_deduplicated = set({edge.uuid: edge.fact for edge in edges}.values())
    assert len(edges_deduplicated) == 1
    assert edges_deduplicated == {'test_entity_1 relates to test_entity_2'}

    edges = await edge_bfs_search(
        graph_driver,
        [entity_node_1.uuid],
        2,
        search_filters,
        group_ids=[group_id],
    )
    edges_deduplicated = set({edge.uuid: edge.fact for edge in edges}.values())
    assert len(edges_deduplicated) == 2
    assert edges_deduplicated == {
        'test_entity_1 relates to test_entity_2',
        'test_entity_2 relates to test_entity_3',
    }


@pytest.mark.asyncio
async def test_node_fulltext_search(
    graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder_client
):
    if graph_driver.provider == GraphProvider.KUZU:
        pytest.skip('Skipping as fulltext indexing not supported for Kuzu')

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )
    await graphiti.build_indices_and_constraints()

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        summary='Summary about Alice',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        summary='Summary about Bob',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)

    # Search for entity edges
    search_filters = SearchFilters(node_labels=['Entity'])
    nodes = await node_fulltext_search(
        graph_driver,
        'Alice',
        search_filters,
        group_ids=[group_id],
    )
    assert len(nodes) == 1
    assert nodes[0].name == entity_node_1.name


@pytest.mark.asyncio
async def test_node_similarity_search(graph_driver, mock_embedder):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_alice',
        summary='Summary about Alice',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_bob',
        summary='Summary about Bob',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)

    # Search for entity edges
    search_filters = SearchFilters(node_labels=['Entity'])
    nodes = await node_similarity_search(
        graph_driver,
        entity_node_1.name_embedding,
        search_filters,
        group_ids=[group_id],
        min_score=0.9,
    )
    assert len(nodes) == 1
    assert nodes[0].name == entity_node_1.name


@pytest.mark.asyncio
async def test_node_bfs_search(graph_driver, mock_embedder):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    # Create episodic nodes
    episodic_node_1 = EpisodicNode(
        name='test_episodic_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
        source=EpisodeType.message,
        source_description='test_source_description',
        content='test_content',
        valid_at=datetime.now(),
    )

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)
    entity_node_3 = EntityNode(
        name='test_entity_3',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_3.generate_name_embedding(mock_embedder)

    # Create entity edges
    entity_edge_1 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        name='RELATES_TO',
        fact='test_entity_1 relates to test_entity_2',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_edge_1.generate_embedding(mock_embedder)
    entity_edge_2 = EntityEdge(
        source_node_uuid=entity_node_2.uuid,
        target_node_uuid=entity_node_3.uuid,
        name='RELATES_TO',
        fact='test_entity_2 relates to test_entity_3',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_edge_2.generate_embedding(mock_embedder)

    # Create episodic to entity edges
    episodic_edge_1 = EpisodicEdge(
        source_node_uuid=episodic_node_1.uuid,
        target_node_uuid=entity_node_1.uuid,
        created_at=datetime.now(),
        group_id=group_id,
    )

    # Save the graph
    await episodic_node_1.save(graph_driver)
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_node_3.save(graph_driver)
    await entity_edge_1.save(graph_driver)
    await entity_edge_2.save(graph_driver)
    await episodic_edge_1.save(graph_driver)

    # Search for entity nodes
    search_filters = SearchFilters(
        node_labels=['Entity'],
    )

    # Test bfs from episodic node

    nodes = await node_bfs_search(
        graph_driver,
        [episodic_node_1.uuid],
        search_filters,
        1,
        group_ids=[group_id],
    )
    nodes_deduplicated = set({node.uuid: node.name for node in nodes}.values())
    assert len(nodes_deduplicated) == 1
    assert nodes_deduplicated == {'test_entity_1'}

    nodes = await node_bfs_search(
        graph_driver,
        [episodic_node_1.uuid],
        search_filters,
        2,
        group_ids=[group_id],
    )
    nodes_deduplicated = set({node.uuid: node.name for node in nodes}.values())
    assert len(nodes_deduplicated) == 2
    assert nodes_deduplicated == {'test_entity_1', 'test_entity_2'}

    # Test bfs from entity node

    nodes = await node_bfs_search(
        graph_driver,
        [entity_node_1.uuid],
        search_filters,
        1,
        group_ids=[group_id],
    )
    nodes_deduplicated = set({node.uuid: node.name for node in nodes}.values())
    assert len(nodes_deduplicated) == 1
    assert nodes_deduplicated == {'test_entity_2'}


@pytest.mark.asyncio
async def test_episode_fulltext_search(
    graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder_client
):
    if graph_driver.provider == GraphProvider.KUZU:
        pytest.skip('Skipping as fulltext indexing not supported for Kuzu')

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )
    await graphiti.build_indices_and_constraints()

    # Create episodic nodes
    episodic_node_1 = EpisodicNode(
        name='test_episodic_1',
        content='test_content',
        created_at=datetime.now(),
        valid_at=datetime.now(),
        group_id=group_id,
        source=EpisodeType.message,
        source_description='Description about Alice',
    )
    episodic_node_2 = EpisodicNode(
        name='test_episodic_2',
        content='test_content_2',
        created_at=datetime.now(),
        valid_at=datetime.now(),
        group_id=group_id,
        source=EpisodeType.message,
        source_description='Description about Bob',
    )

    # Save the graph
    await episodic_node_1.save(graph_driver)
    await episodic_node_2.save(graph_driver)

    # Search for episodic nodes
    search_filters = SearchFilters(node_labels=['Episodic'])
    nodes = await episode_fulltext_search(
        graph_driver,
        'Alice',
        search_filters,
        group_ids=[group_id],
    )
    assert len(nodes) == 1
    assert nodes[0].name == episodic_node_1.name


@pytest.mark.asyncio
async def test_community_fulltext_search(
    graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder_client
):
    if graph_driver.provider == GraphProvider.KUZU:
        pytest.skip('Skipping as fulltext indexing not supported for Kuzu')

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )
    await graphiti.build_indices_and_constraints()

    # Create community nodes
    community_node_1 = CommunityNode(
        name='Alice',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await community_node_1.generate_name_embedding(mock_embedder)
    community_node_2 = CommunityNode(
        name='Bob',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await community_node_2.generate_name_embedding(mock_embedder)

    # Save the graph
    await community_node_1.save(graph_driver)
    await community_node_2.save(graph_driver)

    # Search for community nodes
    nodes = await community_fulltext_search(
        graph_driver,
        'Alice',
        group_ids=[group_id],
    )
    assert len(nodes) == 1
    assert nodes[0].name == community_node_1.name


@pytest.mark.asyncio
async def test_community_similarity_search(
    graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder_client
):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )
    await graphiti.build_indices_and_constraints()

    # Create community nodes
    community_node_1 = CommunityNode(
        name='Alice',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await community_node_1.generate_name_embedding(mock_embedder)
    community_node_2 = CommunityNode(
        name='Bob',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await community_node_2.generate_name_embedding(mock_embedder)

    # Save the graph
    await community_node_1.save(graph_driver)
    await community_node_2.save(graph_driver)

    # Search for community nodes
    nodes = await community_similarity_search(
        graph_driver,
        community_node_1.name_embedding,
        group_ids=[group_id],
        min_score=0.9,
    )
    assert len(nodes) == 1
    assert nodes[0].name == community_node_1.name


@pytest.mark.asyncio
async def test_get_relevant_nodes(
    graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder_client
):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    if graph_driver.provider == GraphProvider.KUZU:
        pytest.skip('Skipping as tests fail on Kuzu')

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )
    await graphiti.build_indices_and_constraints()

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='Alice',
        summary='Alice',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='Bob',
        summary='Bob',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)
    entity_node_3 = EntityNode(
        name='Alice Smith',
        summary='Alice Smith',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_3.generate_name_embedding(mock_embedder)

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_node_3.save(graph_driver)

    # Search for entity nodes
    search_filters = SearchFilters(node_labels=['Entity'])
    nodes = (
        await get_relevant_nodes(
            graph_driver,
            [entity_node_1],
            search_filters,
            min_score=0.9,
        )
    )[0]
    assert len(nodes) == 2
    assert set({node.name for node in nodes}) == {entity_node_1.name, entity_node_3.name}


@pytest.mark.asyncio
async def test_get_relevant_edges_and_invalidation_candidates(
    graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder_client
):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=mock_llm_client,
        embedder=mock_embedder,
        cross_encoder=mock_cross_encoder_client,
    )
    await graphiti.build_indices_and_constraints()

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        summary='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        summary='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)
    entity_node_3 = EntityNode(
        name='test_entity_3',
        summary='test_entity_3',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_3.generate_name_embedding(mock_embedder)

    now = datetime.now()
    created_at = now
    expired_at = now + timedelta(days=6)
    valid_at = now + timedelta(days=2)
    invalid_at = now + timedelta(days=4)

    # Create entity edges
    entity_edge_1 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        name='RELATES_TO',
        fact='Alice',
        created_at=created_at,
        expired_at=expired_at,
        valid_at=valid_at,
        invalid_at=invalid_at,
        group_id=group_id,
    )
    await entity_edge_1.generate_embedding(mock_embedder)
    entity_edge_2 = EntityEdge(
        source_node_uuid=entity_node_2.uuid,
        target_node_uuid=entity_node_3.uuid,
        name='RELATES_TO',
        fact='Bob',
        created_at=created_at,
        expired_at=expired_at,
        valid_at=valid_at,
        invalid_at=invalid_at,
        group_id=group_id,
    )
    await entity_edge_2.generate_embedding(mock_embedder)
    entity_edge_3 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_3.uuid,
        name='RELATES_TO',
        fact='Alice',
        created_at=created_at,
        expired_at=expired_at,
        valid_at=valid_at,
        invalid_at=invalid_at,
        group_id=group_id,
    )
    await entity_edge_3.generate_embedding(mock_embedder)

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_node_3.save(graph_driver)
    await entity_edge_1.save(graph_driver)
    await entity_edge_2.save(graph_driver)
    await entity_edge_3.save(graph_driver)

    # Search for entity nodes
    search_filters = SearchFilters(
        node_labels=['Entity'],
        edge_types=['RELATES_TO'],
        created_at=[
            [DateFilter(date=created_at, comparison_operator=ComparisonOperator.equals)],
        ],
        expired_at=[
            [DateFilter(date=now, comparison_operator=ComparisonOperator.not_equals)],
        ],
        valid_at=[
            [
                DateFilter(
                    date=now + timedelta(days=1),
                    comparison_operator=ComparisonOperator.greater_than_equal,
                )
            ],
            [
                DateFilter(
                    date=now + timedelta(days=3),
                    comparison_operator=ComparisonOperator.less_than_equal,
                )
            ],
        ],
        invalid_at=[
            [
                DateFilter(
                    date=now + timedelta(days=3),
                    comparison_operator=ComparisonOperator.greater_than,
                )
            ],
            [
                DateFilter(
                    date=now + timedelta(days=5), comparison_operator=ComparisonOperator.less_than
                )
            ],
        ],
    )
    edges = (
        await get_relevant_edges(
            graph_driver,
            [entity_edge_1],
            search_filters,
            min_score=0.9,
        )
    )[0]
    assert len(edges) == 1
    assert set({edge.name for edge in edges}) == {entity_edge_1.name}

    edges = (
        await get_edge_invalidation_candidates(
            graph_driver,
            [entity_edge_1],
            search_filters,
            min_score=0.9,
        )
    )[0]
    assert len(edges) == 2
    assert set({edge.name for edge in edges}) == {entity_edge_1.name, entity_edge_3.name}


@pytest.mark.asyncio
async def test_node_distance_reranker(graph_driver, mock_embedder):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)
    entity_node_3 = EntityNode(
        name='test_entity_3',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_3.generate_name_embedding(mock_embedder)

    # Create entity edges
    entity_edge_1 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        name='RELATES_TO',
        fact='test_entity_1 relates to test_entity_2',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_edge_1.generate_embedding(mock_embedder)

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_node_3.save(graph_driver)
    await entity_edge_1.save(graph_driver)

    # Test reranker
    reranked_uuids, reranked_scores = await node_distance_reranker(
        graph_driver,
        [entity_node_2.uuid, entity_node_3.uuid],
        entity_node_1.uuid,
    )
    uuid_to_name = {
        entity_node_1.uuid: entity_node_1.name,
        entity_node_2.uuid: entity_node_2.name,
        entity_node_3.uuid: entity_node_3.name,
    }
    names = [uuid_to_name[uuid] for uuid in reranked_uuids]
    assert names == [entity_node_2.name, entity_node_3.name]
    assert np.allclose(reranked_scores, [1.0, 0.0])


@pytest.mark.asyncio
async def test_episode_mentions_reranker(graph_driver, mock_embedder):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    # Create episodic nodes
    episodic_node_1 = EpisodicNode(
        name='test_episodic_1',
        content='test_content',
        created_at=datetime.now(),
        valid_at=datetime.now(),
        group_id=group_id,
        source=EpisodeType.message,
        source_description='Description about Alice',
    )

    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)

    # Create entity edges
    episodic_edge_1 = EpisodicEdge(
        source_node_uuid=episodic_node_1.uuid,
        target_node_uuid=entity_node_1.uuid,
        created_at=datetime.now(),
        group_id=group_id,
    )

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await episodic_node_1.save(graph_driver)
    await episodic_edge_1.save(graph_driver)

    # Test reranker
    reranked_uuids, reranked_scores = await episode_mentions_reranker(
        graph_driver,
        [[entity_node_1.uuid, entity_node_2.uuid]],
    )
    uuid_to_name = {entity_node_1.uuid: entity_node_1.name, entity_node_2.uuid: entity_node_2.name}
    names = [uuid_to_name[uuid] for uuid in reranked_uuids]
    assert names == [entity_node_1.name, entity_node_2.name]
    assert np.allclose(reranked_scores, [1.0, float('inf')])


@pytest.mark.asyncio
async def test_get_embeddings_for_edges(graph_driver, mock_embedder):
    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)
    entity_node_2 = EntityNode(
        name='test_entity_2',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_2.generate_name_embedding(mock_embedder)

    # Create entity edges
    entity_edge_1 = EntityEdge(
        source_node_uuid=entity_node_1.uuid,
        target_node_uuid=entity_node_2.uuid,
        name='RELATES_TO',
        fact='test_entity_1 relates to test_entity_2',
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_edge_1.generate_embedding(mock_embedder)

    # Save the graph
    await entity_node_1.save(graph_driver)
    await entity_node_2.save(graph_driver)
    await entity_edge_1.save(graph_driver)

    # Get embeddings for edges
    embeddings = await get_embeddings_for_edges(graph_driver, [entity_edge_1])
    assert len(embeddings) == 1
    assert entity_edge_1.uuid in embeddings
    assert np.allclose(embeddings[entity_edge_1.uuid], entity_edge_1.fact_embedding)


@pytest.mark.asyncio
async def test_get_embeddings_for_nodes(graph_driver, mock_embedder):
    # Create entity nodes
    entity_node_1 = EntityNode(
        name='test_entity_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await entity_node_1.generate_name_embedding(mock_embedder)

    # Save the graph
    await entity_node_1.save(graph_driver)

    # Get embeddings for edges
    embeddings = await get_embeddings_for_nodes(graph_driver, [entity_node_1])
    assert len(embeddings) == 1
    assert entity_node_1.uuid in embeddings
    assert np.allclose(embeddings[entity_node_1.uuid], entity_node_1.name_embedding)


@pytest.mark.asyncio
async def test_get_embeddings_for_communities(graph_driver, mock_embedder):
    # Create community nodes
    community_node_1 = CommunityNode(
        name='test_community_1',
        labels=[],
        created_at=datetime.now(),
        group_id=group_id,
    )
    await community_node_1.generate_name_embedding(mock_embedder)

    # Save the graph
    await community_node_1.save(graph_driver)

    # Get embeddings for communities
    embeddings = await get_embeddings_for_communities(graph_driver, [community_node_1])
    assert len(embeddings) == 1
    assert community_node_1.uuid in embeddings
    assert np.allclose(embeddings[community_node_1.uuid], community_node_1.name_embedding)
