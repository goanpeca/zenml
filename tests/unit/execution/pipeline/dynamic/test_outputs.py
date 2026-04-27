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
"""Tests for dynamic output future helpers."""

from concurrent.futures import Future
from types import SimpleNamespace
from uuid import uuid4

import pytest

from zenml.execution.pipeline.dynamic.outputs import PipelineFuture
from zenml.models import ArtifactVersionResponse


def _artifact_response(name: str) -> ArtifactVersionResponse:
    """Creates an artifact response test double."""
    return ArtifactVersionResponse.model_construct(
        id=uuid4(),
        body=None,
        metadata=None,
        resources=None,
        name=name,
    )


def test_pipeline_future_exposes_outputs_by_name_and_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests PipelineFuture output access helpers."""
    first = _artifact_response("first")
    second = _artifact_response("second")
    wrapped: Future[SimpleNamespace] = Future()
    wrapped.set_result(
        SimpleNamespace(
            outputs={
                "alpha": first,
                "beta": second,
            }
        )
    )

    future = PipelineFuture(
        invocation_id="pipeline:child",
        declared_output_names=["alpha", "beta"],
    )
    future._set_startup_result(wrapped)

    assert future.get_artifact("alpha") is first
    assert future.get_artifact("beta") is second
    assert future[0] is first
    assert future[1] is second
    assert future[:] == (first, second)
    assert len(future) == 2
    assert tuple(future) == (first, second)


def test_pipeline_future_handles_missing_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests missing output lookup on PipelineFuture."""
    artifact = _artifact_response("only")
    wrapped: Future[SimpleNamespace] = Future()
    wrapped.set_result(SimpleNamespace(outputs={"only": artifact}))

    future = PipelineFuture(
        invocation_id="pipeline:child",
        declared_output_names=["only"],
    )
    future._set_startup_result(wrapped)

    with pytest.raises(KeyError):
        future.get_artifact("missing")


def test_pipeline_future_cancel_startup_propagates_to_result() -> None:
    """Tests that cancelling startup surfaces the failure on result()."""
    future = PipelineFuture(
        invocation_id="pipeline:child",
        declared_output_names=["only"],
    )
    future._cancel_startup(RuntimeError("startup cancelled"))

    with pytest.raises(RuntimeError, match="startup cancelled"):
        future.result()


def test_pipeline_future_wait_raises_execution_failure() -> None:
    """Tests PipelineFuture.wait surfaces execution failures."""
    wrapped: Future[SimpleNamespace] = Future()
    wrapped.set_exception(RuntimeError("child failed"))

    future = PipelineFuture(
        invocation_id="pipeline:child",
        declared_output_names=["only"],
    )
    future._set_startup_result(wrapped)

    with pytest.raises(RuntimeError, match="child failed"):
        future.wait()
