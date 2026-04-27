#  Copyright (c) ZenML GmbH 2026. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
"""Tests for dynamic pipeline input resolution helpers."""

from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from zenml.execution.pipeline.dynamic.inputs import (
    await_step_inputs,
    collect_upstream_node_ids,
)
from zenml.execution.pipeline.dynamic.outputs import PipelineFuture
from zenml.models import ArtifactVersionResponse


def _artifact_response(name: str = "artifact") -> ArtifactVersionResponse:
    """Build a minimal artifact response double."""
    return ArtifactVersionResponse.model_construct(
        id=uuid4(),
        body=None,
        metadata=None,
        resources=None,
        name=name,
    )


def test_await_step_inputs_resolves_single_output_pipeline_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests `await_step_inputs` resolves single-output sub-pipeline futures."""
    wrapped: Future[Any] = Future()
    artifact = _artifact_response()
    wrapped.set_result(SimpleNamespace(outputs={"output": artifact}))

    pipeline_future = PipelineFuture(
        invocation_id="pipeline:child",
        declared_output_names=["output"],
    )
    pipeline_future._set_startup_result(wrapped)

    inputs = await_step_inputs({"name": pipeline_future})

    assert inputs["name"] == artifact


def test_await_step_inputs_rejects_multi_output_pipeline_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests `await_step_inputs` rejects multi-output sub-pipeline futures."""
    wrapped: Future[Any] = Future()
    first = _artifact_response("one")
    second = _artifact_response("two")
    wrapped.set_result(
        SimpleNamespace(outputs={"a": first.id, "b": second.id})
    )
    fake_client = SimpleNamespace(
        get_artifact_version=lambda artifact_id: {
            first.id: first,
            second.id: second,
        }[artifact_id]
    )
    monkeypatch.setattr(
        "zenml.execution.pipeline.dynamic.utils.Client",
        lambda: fake_client,
    )

    pipeline_future = PipelineFuture(
        invocation_id="pipeline:child",
        declared_output_names=["a", "b"],
    )
    pipeline_future._set_startup_result(wrapped)

    with pytest.raises(RuntimeError, match="multiple output artifacts"):
        await_step_inputs({"name": pipeline_future})


def test_collect_upstream_node_ids_accepts_pipeline_future_after() -> None:
    """Tests `collect_upstream_node_ids` reads PipelineFuture invocations."""
    pipeline_future = PipelineFuture(
        declared_output_names=["output"],
        invocation_id="pipeline:child",
    )

    upstream_ids = collect_upstream_node_ids(
        inputs={},
        after=pipeline_future,
    )

    assert upstream_ids == ["pipeline:child"]
