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
"""Tests for sub-pipeline lifecycle on `DynamicPipelineRunner`."""

import threading
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any, Callable, Optional
from unittest.mock import Mock
from uuid import uuid4

import pytest

from zenml.execution.pipeline.dynamic import runner as runner_module
from zenml.execution.pipeline.dynamic.runner import DynamicPipelineRunner
from zenml.models import ArtifactVersionResponse


class _CapturingExecutor:
    """Executor that captures submitted work without executing it."""

    def __init__(self) -> None:
        """Initialize the executor."""
        self.submitted_fn: Optional[Callable[..., Any]] = None
        self.submitted_args: tuple[Any, ...] = ()
        self.submitted_kwargs: dict[str, Any] = {}
        self.future: Future[Any] = Future()

    def submit(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Future[Any]:
        """Capture a submitted call.

        Args:
            fn: Submitted callable.
            *args: Positional callable args.
            **kwargs: Keyword callable args.

        Returns:
            A placeholder future.
        """
        self.submitted_fn = fn
        self.submitted_args = args
        self.submitted_kwargs = kwargs
        return self.future


def _build_runner(
    executor: Optional[_CapturingExecutor] = None,
) -> DynamicPipelineRunner:
    """Build a `DynamicPipelineRunner` instance for sub-pipeline tests.

    Bypasses `__init__` (which requires a real stack/snapshot) and sets only
    the attributes the sub-pipeline lifecycle methods touch. Methods that the
    sub-pipeline path normally delegates to (graph + future registry updates,
    invocation ID allocation, etc.) are replaced with mocks.

    Args:
        executor: Optional executor implementation.

    Returns:
        A bare runner instance suitable for invoking sub-pipeline methods on.
    """
    runner = DynamicPipelineRunner.__new__(DynamicPipelineRunner)
    runner._snapshot = SimpleNamespace(  # type: ignore[assignment]
        pipeline=SimpleNamespace(name="parent")
    )
    runner._run = SimpleNamespace(id="parent-run-id")  # type: ignore[assignment]
    runner._orchestrator = object()  # type: ignore[assignment]
    runner._orchestrator_run_id = "parent-orc-id"
    runner._executor = executor or _CapturingExecutor()  # type: ignore[assignment]
    runner._lifecycle_lock = threading.RLock()
    runner._existing_children_lock = threading.Lock()
    runner._existing_children_cache = None
    runner._dependency_graph = Mock()
    runner._future_registry = Mock()
    runner._failure_detected = False
    runner._child_runners = {}
    runner.mark_node_starting = Mock()  # type: ignore[method-assign]
    runner.mark_node_running = Mock()  # type: ignore[method-assign]
    runner.mark_node_succeeded = Mock()  # type: ignore[method-assign]
    runner.mark_node_failed = Mock()  # type: ignore[method-assign]
    runner.record_failure = Mock()  # type: ignore[method-assign]
    runner.raise_if_startup_cancelled = Mock()  # type: ignore[method-assign]
    runner.allocate_invocation_id = Mock(  # type: ignore[method-assign]
        side_effect=lambda base_name: base_name
    )
    runner.notify_graph_changed = Mock()  # type: ignore[method-assign]
    return runner


def _create_compiled_child_run() -> SimpleNamespace:
    """Create a child run payload with the fields used by the runner."""
    return SimpleNamespace(
        id="child-run-id",
        snapshot=SimpleNamespace(
            pipeline=SimpleNamespace(name="child-pipeline")
        ),
        orchestrator_run_id="orc-id",
    )


def _artifact_response(name: str = "artifact") -> ArtifactVersionResponse:
    """Build a minimal artifact response double."""
    return ArtifactVersionResponse.model_construct(
        id=uuid4(),
        body=None,
        metadata=None,
        resources=None,
        name=name,
    )


def test_handle_subpipeline_ready_marks_starting_before_compile() -> None:
    """Tests sub-pipeline nodes are marked starting before compilation."""
    runner = _build_runner()
    call_order: list[str] = []

    def _record_starting(node_id: str) -> None:
        assert node_id == "pipeline:test"
        call_order.append("mark_starting")

    def _compile(*_: Any, **__: Any) -> Any:
        call_order.append("compile_child_run")
        return _create_compiled_child_run()

    runner.mark_node_starting = _record_starting  # type: ignore[method-assign]
    runner._prepare_child_run = _compile  # type: ignore[method-assign]
    runner._build_child_runner = Mock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            run_pipeline=lambda: None,
            run=SimpleNamespace(id="child"),
        )
    )

    node = SimpleNamespace(
        node_id="pipeline:test",
        pipeline=object(),
        args=(),
        kwargs={},
    )
    runner._handle_subpipeline_ready(node=node)

    assert call_order[:2] == ["mark_starting", "compile_child_run"]


def test_handle_subpipeline_ready_marks_failure_for_unsuccessful_child() -> (
    None
):
    """Tests the launch closure surfaces non-success child runs as failures."""
    runner = _build_runner()
    runner._prepare_child_run = Mock(  # type: ignore[method-assign]
        return_value=_create_compiled_child_run()
    )
    failed_run = SimpleNamespace(
        id="child",
        name="child-pipeline-run",
        status=SimpleNamespace(
            is_successful=False, is_finished=True, value="failed"
        ),
        exception_info=None,
    )
    runner._build_child_runner = Mock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(run_pipeline=lambda: None, run=failed_run)
    )
    captured_future: dict[str, Any] = {}

    def _capture(node_id: str, execution_future: Any) -> None:
        captured_future["future"] = execution_future

    runner._handle_subpipeline_startup_succeeded = _capture  # type: ignore[method-assign]

    node = SimpleNamespace(
        node_id="pipeline:test",
        pipeline=object(),
        args=(),
        kwargs={},
    )
    runner._handle_subpipeline_ready(node=node)

    # Sub-pipelines run on a dedicated daemon thread; wait for it to finish
    # by reading the bound execution future.
    with pytest.raises(RuntimeError, match="failed with status"):
        captured_future["future"].result(timeout=5)

    runner.mark_node_failed.assert_called_once()  # type: ignore[attr-defined]
    runner.record_failure.assert_called_once()  # type: ignore[attr-defined]
    runner.mark_node_succeeded.assert_not_called()  # type: ignore[attr-defined]


def test_handle_subpipeline_ready_cleans_up_placeholder_on_cancel() -> None:
    """Tests placeholder runs are marked failed if cancellation races compile.

    Models the real race: the early cancellation check passes, then a parent
    failure is recorded while compile is running, and the lock-protected
    recheck must mark the placeholder failed and raise.
    """
    runner = _build_runner()
    runner._build_child_runner = Mock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            run_pipeline=lambda: None,
            run=SimpleNamespace(id="child"),
        )
    )

    def _compile_then_flip_failure(*_: Any, **__: Any) -> SimpleNamespace:
        runner._failure_detected = True
        return _create_compiled_child_run()

    runner._prepare_child_run = _compile_then_flip_failure  # type: ignore[method-assign]

    cleanup_calls: list[str] = []

    def _cleanup(child_run: Any, reason: str) -> None:
        cleanup_calls.append(reason)

    runner._mark_subpipeline_placeholder_failed = _cleanup  # type: ignore[method-assign]

    def _raise_if_failed() -> None:
        if runner._failure_detected:
            raise RuntimeError("startup cancelled")

    runner.raise_if_startup_cancelled = _raise_if_failed  # type: ignore[method-assign]

    node = SimpleNamespace(
        node_id="pipeline:test",
        pipeline=object(),
        args=(),
        kwargs={},
    )
    with pytest.raises(RuntimeError, match="startup cancelled"):
        runner._handle_subpipeline_ready(node=node)

    assert cleanup_calls and "shut down" in cleanup_calls[0]


def test_compile_child_run_reuses_existing_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests `_compile_child_run` reuses existing child runs."""
    runner = _build_runner()
    existing_run = SimpleNamespace(id="existing", snapshot=object())
    runner._existing_child_run = Mock(return_value=existing_run)  # type: ignore[method-assign]

    create_placeholder_run_mock = Mock()
    monkeypatch.setattr(
        runner_module,
        "create_placeholder_run",
        create_placeholder_run_mock,
    )

    pipeline = SimpleNamespace(name="child", entrypoint=lambda: None)
    result = runner._prepare_child_run(
        pipeline=pipeline,
        args=(),
        kwargs={},
        after=None,
        child_invocation_id="pipeline:child",
    )

    assert result is existing_run
    create_placeholder_run_mock.assert_not_called()


def test_submit_subpipeline_synchronous_loads_outputs_from_run() -> None:
    """Tests sync submit returns persisted outputs from the runner's run."""
    runner = _build_runner()
    runner._prepare_child_run = Mock(  # type: ignore[method-assign]
        return_value=_create_compiled_child_run()
    )
    artifact = _artifact_response()
    finished_run = SimpleNamespace(
        id="child-run-id",
        name="child-pipeline-run",
        status=SimpleNamespace(is_successful=True, is_finished=True),
        outputs={"result": artifact},
        exception_info=None,
    )
    runner._build_child_runner = Mock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            run_pipeline=lambda: None, run=finished_run
        )
    )

    pipeline = SimpleNamespace(
        name="child",
        copy=lambda: SimpleNamespace(
            name="child", entrypoint=lambda: None, copy=lambda: None
        ),
        entrypoint=lambda: None,
    )

    result = runner.submit_subpipeline(
        pipeline=pipeline, args=(), kwargs={}, concurrent=False
    )
    assert result == artifact


def test_submit_subpipeline_synchronous_raises_for_unsuccessful_child() -> (
    None
):
    """Tests sync submit raises if the child run did not succeed."""
    runner = _build_runner()
    runner._prepare_child_run = Mock(  # type: ignore[method-assign]
        return_value=_create_compiled_child_run()
    )
    failed_run = SimpleNamespace(
        id="child-run-id",
        name="child-pipeline-run",
        status=SimpleNamespace(
            is_successful=False, is_finished=True, value="failed"
        ),
        outputs={},
        exception_info=None,
    )
    runner._build_child_runner = Mock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(run_pipeline=lambda: None, run=failed_run)
    )

    pipeline = SimpleNamespace(
        name="child",
        copy=lambda: SimpleNamespace(
            name="child", entrypoint=lambda: None, copy=lambda: None
        ),
        entrypoint=lambda: None,
    )

    with pytest.raises(RuntimeError, match="failed with status"):
        runner.submit_subpipeline(
            pipeline=pipeline, args=(), kwargs={}, concurrent=False
        )
