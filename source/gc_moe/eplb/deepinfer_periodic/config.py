"""Validated experiment configuration and DeepInfer's rearrangement clock."""

from dataclasses import dataclass
from typing import Mapping, Optional


def resolve_num_extra_experts(logical: int, minimum: int, ep_size: int) -> int:
    if logical <= 0 or minimum < 0 or ep_size <= 0:
        raise ValueError('Invalid logical expert count, redundant floor, or EP size')
    return minimum + (-(logical + minimum) % ep_size)


def initial_layout(logical: int, extra: int) -> list[int]:
    if logical <= 0 or extra < 0:
        raise ValueError('Invalid expert count')
    return list(range(logical)) + [index % logical for index in range(extra)]


@dataclass(frozen=True)
class Settings:
    phase: str = 'decode'
    window: int = 1000
    interval: int = 10000
    minimum_extra: int = 1
    p2p_groups: int = 1
    stat_interval: int = 128
    expected_batch: int = 32
    first_rearrangement: Optional[int] = None
    collect_steps: Optional[int] = None
    max_rearrangements: Optional[int] = None

    def __post_init__(self):
        if self.phase not in ('prefill', 'decode'):
            raise ValueError('EPLB phase must be prefill or decode')
        if self.window < 0 or self.interval < 0:
            raise ValueError('Window and interval must be non-negative')
        if self.interval and not 0 < self.window <= self.interval:
            raise ValueError('Enabled rearrangement requires 1 <= window <= interval')
        if self.minimum_extra < 0 or min(self.p2p_groups, self.stat_interval, self.expected_batch) <= 0:
            raise ValueError('Invalid redundancy, P2P, statistics, or batch setting')
        if self.first_rearrangement is not None and not 0 < self.first_rearrangement <= self.interval:
            raise ValueError('First rearrangement must be within the enabled interval')
        if self.collect_steps is not None and self.collect_steps < 1:
            raise ValueError('Collection limit must be positive')
        if self.max_rearrangements is not None and self.max_rearrangements < 0:
            raise ValueError('Rearrangement limit must be non-negative')

    @classmethod
    def from_env(cls, env: Mapping[str, str]):
        phase = env.get('RTP_DEEPINFER_EPLB_PHASE', 'decode')
        defaults = (100, 1000, 4) if phase == 'prefill' else (1000, 10000, 1)
        return cls(
            phase=phase,
            window=int(env.get('DEEPINFER_EXPERT_COUNT_WINDOW_SIZE', defaults[0])),
            interval=int(env.get('DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL', defaults[1])),
            minimum_extra=int(env.get('DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS', defaults[2])),
            p2p_groups=int(env.get('DEEPINFER_REARRANGE_P2P_GROUPS', '1')),
            stat_interval=int(env.get('DEEPINFER_MOE_LB_STAT_INTERVAL', '128')),
            expected_batch=int(env.get('RTP_DEEPINFER_EPLB_BATCH', '32')),
            first_rearrangement=int(env['RTP_EPLB_FIRST_REARRANGEMENT']) if env.get('RTP_EPLB_FIRST_REARRANGEMENT') else None,
            collect_steps=int(env['RTP_EPLB_COLLECT_STEPS']) if env.get('RTP_EPLB_COLLECT_STEPS') else None,
            max_rearrangements=int(env['RTP_EPLB_MAX_REARRANGEMENTS']) if env.get('RTP_EPLB_MAX_REARRANGEMENTS') else None,
        )


class RearrangementClock:
    def __init__(self, interval: int, first_at: Optional[int] = None):
        if interval < 0:
            raise ValueError('Interval must be non-negative')
        self.interval = interval
        if first_at is not None and not 0 < first_at <= interval:
            raise ValueError('Invalid initial rearrangement delay')
        delay = first_at if first_at is not None else max(1, interval // 4)
        self.position = max(0, interval - delay)
        self.completed_steps = 0

    def tick(self) -> bool:
        self.completed_steps += 1
        if not self.interval:
            return False
        self.position += 1
        if self.position < self.interval:
            return False
        self.position = 0
        return True


def chunk_ranges(local_tokens: int, global_tokens: int, capacity: int):
    if not 0 <= local_tokens <= global_tokens or capacity <= 0:
        raise ValueError('Invalid synchronized prefill chunk dimensions')
    return [(min(offset, local_tokens), min(offset + capacity, local_tokens))
            for offset in range(0, max(1, global_tokens), capacity)]
