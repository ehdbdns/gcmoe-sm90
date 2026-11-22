"""Exclusive elapsed and caller-thread CPU time on the in-process decode path."""
from functools import wraps
from threading import get_ident
from time import perf_counter_ns, thread_time_ns
from unittest.mock import patch


def install(stack, engine, steps):
    owner = get_ident()
    nesting = []
    rows = []
    instrumented_futures = set()

    def timed(fn, name):
        @wraps(fn)
        def call(*args, **kwargs):
            if get_ident() != owner or (not nesting and name != 'step'):
                return fn(*args, **kwargs)
            child = [0, 0]
            nesting.append(child)
            start, cpu = perf_counter_ns(), thread_time_ns()
            try:
                value = fn(*args, **kwargs)
            finally:
                elapsed_cpu = thread_time_ns() - cpu
                elapsed = perf_counter_ns() - start
                nesting.pop()
                if nesting:
                    nesting[-1][0] += elapsed
                    nesting[-1][1] += elapsed_cpu
                rows.append((len(steps), name, elapsed-child[0], elapsed_cpu-child[1]))
            return value
        return call

    client = engine.engine_core
    core = client.engine_core
    executor = core.model_executor
    for obj, names in (
        (engine, ('step',)),
        (client, ('get_output', 'abort_requests')),
        (core, ('step_fn', 'post_step')),
        (core.scheduler, ('schedule', 'get_grammar_bitmask', 'update_from_output')),
        (executor, ('execute_model', 'sample_tokens')),
        (engine.output_processor, ('process_outputs', 'update_scheduler_stats')),
    ):
        for name in names:
            fn = timed(getattr(obj, name), name)
            if obj is executor:
                # Synchronous sampling returns model output. Async execution
                # returns a Future; time the actual consumer's result() call.
                def submit(*args, _fn=fn, **kwargs):
                    value = _fn(*args, **kwargs)
                    cls = type(value)
                    if cls not in instrumented_futures and callable(getattr(cls, 'result', None)):
                        stack.enter_context(patch.object(cls, 'result', timed(cls.result, 'worker_result')))
                        instrumented_futures.add(cls)
                    return value
                fn = submit
            stack.enter_context(patch.object(obj, name, fn))
    return rows
