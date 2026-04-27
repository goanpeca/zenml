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
"""Tests for dynamic future registry pipeline tracking."""

from concurrent.futures import Future

from zenml.execution.pipeline.dynamic.future_registry import FutureRegistry
from zenml.execution.pipeline.dynamic.outputs import PipelineFuture


def test_pipeline_future_tracked_until_execution_completes() -> None:
    """Tests that a registered pipeline future contributes to in-progress work.

    The registry should report in-progress work for as long as the bound
    execution future is still running.
    """
    registry = FutureRegistry()
    pipeline_future = PipelineFuture(
        invocation_id="pipeline:child",
        declared_output_names=["result"],
    )
    wrapped: Future[object] = Future()

    registry.register_pipeline_future(
        node_id="pipeline:child", future=pipeline_future
    )
    pipeline_future._set_startup_result(wrapped)

    assert registry.has_in_progress_work()
    wrapped.set_result(object())
    assert not registry.has_in_progress_work()
