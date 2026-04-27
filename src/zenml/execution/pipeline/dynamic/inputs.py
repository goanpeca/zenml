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
"""Input resolution helpers for dynamic pipeline execution."""

import inspect
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Sequence,
    Tuple,
    Union,
)

from zenml.execution.pipeline.dynamic.outputs import (
    AnyOutputFuture,
    ArtifactFuture,
    BaseStepFuture,
    MapResultsFuture,
    PipelineFuture,
    StepFuture,
)
from zenml.execution.pipeline.dynamic.utils import collect_futures


def convert_to_keyword_arguments(
    func: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    apply_defaults: bool = False,
) -> Dict[str, Any]:
    """Convert function arguments to keyword arguments.

    Args:
        func: The function to convert the arguments to keyword arguments for.
        args: The arguments to convert to keyword arguments.
        kwargs: The keyword arguments to convert to keyword arguments.
        apply_defaults: Whether to apply the function default values.

    Returns:
        The keyword arguments.
    """
    signature = inspect.signature(func, follow_wrapped=True)
    bound_args = signature.bind_partial(*args, **kwargs)
    if apply_defaults:
        bound_args.apply_defaults()

    return bound_args.arguments


def await_step_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Await the inputs of a step.

    Args:
        inputs: The inputs of the step.

    Raises:
        RuntimeError: If a step run future or a sub-pipeline future referring
            to multiple output artifacts is passed as an input.

    Returns:
        The awaited inputs.
    """
    result = {}
    for key, value in inputs.items():
        if isinstance(value, MapResultsFuture):
            value = value.futures

        if (
            isinstance(value, (list, tuple))
            and value
            and all(isinstance(item, StepFuture) for item in value)
        ):
            if any(len(item._output_keys) != 1 for item in value):
                raise RuntimeError(
                    f"Invalid step input `{key}`: Passing a future that refers "
                    "to multiple output artifacts as an input to another step "
                    "is not allowed."
                )
            value = [item.artifacts() for item in value]
        elif isinstance(value, StepFuture):
            if len(value._output_keys) != 1:
                raise RuntimeError(
                    f"Invalid step input `{key}`: Passing a future that refers "
                    "to multiple output artifacts as an input to another step "
                    "is not allowed."
                )
            value = value.artifacts()
        elif (
            isinstance(value, (list, tuple))
            and value
            and all(isinstance(item, PipelineFuture) for item in value)
        ):
            if any(len(item) != 1 for item in value):
                raise RuntimeError(
                    f"Invalid step input `{key}`: Passing a sub-pipeline "
                    "future that refers to multiple output artifacts as an "
                    "input to another step is not allowed."
                )
            value = [item.artifacts() for item in value]
        elif isinstance(value, PipelineFuture):
            if len(value) != 1:
                raise RuntimeError(
                    f"Invalid step input `{key}`: Passing a sub-pipeline "
                    "future that refers to multiple output artifacts as an "
                    "input to another step is not allowed."
                )
            value = value.artifacts()

        if (
            isinstance(value, (list, tuple))
            and value
            and all(isinstance(item, ArtifactFuture) for item in value)
        ):
            value = [item.result() for item in value]

        if isinstance(value, ArtifactFuture):
            value = value.result()

        result[key] = value

    return result


def collect_upstream_node_ids(
    inputs: Dict[str, Any],
    after: Union["AnyOutputFuture", Sequence["AnyOutputFuture"], None],
) -> List[str]:
    """Collect upstream node IDs from step inputs and `after` futures.

    Args:
        inputs: The step inputs.
        after: Optional upstream futures for explicit ordering.

    Returns:
        The upstream node IDs.
    """
    return [
        future.invocation_id
        for future in collect_futures(inputs=inputs, after=after)
    ]


def get_running_upstream_dependencies(
    inputs: Dict[str, Any],
    after: Union["AnyOutputFuture", Sequence["AnyOutputFuture"], None],
) -> List[str]:
    """Get all running upstream dependencies for a step.

    Args:
        inputs: The inputs of the step.
        after: The step run futures to wait for.

    Raises:
        TypeError: If an unexpected future type is passed.

    Returns:
        The list of running upstream dependencies.
    """
    futures = collect_futures(inputs=inputs, after=after)

    dependencies = []

    for future in futures:
        if isinstance(future, MapResultsFuture):
            if future.startup_succeeded:
                for item in future.futures:
                    if item.running():
                        dependencies.append(item.invocation_id)
            elif future.running():
                dependencies.append(future.invocation_id)
        elif isinstance(future, BaseStepFuture):
            if future.running():
                dependencies.append(future.invocation_id)
        elif isinstance(future, PipelineFuture):
            if future.running():
                dependencies.append(future.invocation_id)
        else:
            raise TypeError(f"Unexpected future type: {type(future)}")

    return dependencies
