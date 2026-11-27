"""Transactional numerical and categorical choices with task-wide preferences."""

import copy
import pickle
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np

from r0.modules.priors.lib.categorical_preferences import CategoricalDraw, CategoricalParameter
from r0.modules.priors.lib.categorical_sampling import CategoricalSampler
from r0.modules.priors.lib.joint_sampling_records import (
    JointDecision,
    JointSamplingRecord,
    check_numerical_contract,
    numerical_policy,
    validate_path,
)
from r0.modules.priors.lib.parameter_preferences import (
    NumericalDraw,
    NumericalParameter,
    ParameterSampler,
)
from r0.modules.priors.lib.preference_limits import PreferenceLimits

_DEFAULT_LIMITS = PreferenceLimits()


class JointSamplingContext:
    """One outer scope; categorical weight hyperpriors retain their nested scopes."""

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        mode: str = "meta",
        categorical_policy: str = "shared",
        global_policy: str = "scalar-quantile-v1",
        limits: PreferenceLimits = _DEFAULT_LIMITS,
    ) -> None:
        self._numerical = ParameterSampler(
            rng, default_mode=mode, global_policy=numerical_policy(global_policy), limits=limits
        )
        self._categorical = CategoricalSampler(
            rng, default_mode=mode, policy=categorical_policy, limits=limits
        )
        self._rng = rng
        self._events: list[JointDecision] = []
        self._path: tuple[str, ...] = ()
        self._closed = False
        self._busy = False
        self._completed: set[tuple[str, ...]] = set()
        self._roles: set[tuple[tuple[str, ...], str]] = set()
        self._counts = {"numerical": 0, "categorical": 0, "forced_identity": 0}
        self._mode = mode
        self._global_policy = global_policy
        self._contracts: dict[tuple[str, str], NumericalParameter] = {}
        self._limits = limits

    def _available(self) -> None:
        if self._closed:
            raise ValueError("joint operation handle is closed")
        if self._busy:
            raise ValueError("joint context has an active child transaction")

    @contextmanager
    def transaction(self, name: str) -> Generator["JointSamplingContext"]:
        self._available()
        path = (*self._path, name)
        validate_path(path)
        if path in self._completed:
            raise ValueError("joint operation has already completed")
        staged = copy.copy(self)
        staged._rng = copy.deepcopy(self._rng)
        staged._numerical = self._numerical.fork(staged._rng)
        staged._categorical = self._categorical.fork(staged._rng)
        staged._events = self._events.copy()
        staged._path = path
        staged._completed = self._completed.copy()
        staged._roles = self._roles.copy()
        staged._counts = self._counts.copy()
        staged._contracts = self._contracts.copy()
        # Only serialize trusted in-memory state; never load a caller pickle.
        initial_state = pickle.dumps(self._rng.bit_generator.state)
        self._busy = True
        try:
            yield staged
            staged.freeze()
            if pickle.dumps(self._rng.bit_generator.state) != initial_state:
                raise ValueError("caller RNG changed during the joint transaction")
            self._rng.bit_generator.state = staged._rng.bit_generator.state
            self._numerical = staged._numerical.fork(self._rng)
            self._categorical = staged._categorical.fork(self._rng)
            self._events = staged._events.copy()
            self._completed = staged._completed.copy()
            self._roles = staged._roles.copy()
            self._counts = staged._counts.copy()
            self._contracts = staged._contracts.copy()
            # Ancestor operations are represented by any owned descendant event.
            if any(event.operation[: len(path)] == path for event in staged._events):
                self._completed.add(path)
        finally:
            staged._closed = True
            self._busy = False

    def _event(self, role: str, kind: str, reason: str | None = None) -> JointDecision:
        self._available()
        index = None if kind == "forced_identity" else self._counts[kind]
        event = JointDecision(self._path, role, kind, index, reason)
        if (self._path, role) in self._roles:
            raise ValueError("joint decision role already belongs to an earlier draw")
        return event

    def _accept(self, event: JointDecision, rng: np.random.Generator | None = None) -> None:
        if rng is not None:
            self._rng.bit_generator.state = rng.bit_generator.state
        self._events.append(event)
        self._roles.add((event.operation, event.role))
        self._counts[event.kind] += 1

    def _remaining(self, other: dict[str, int]) -> PreferenceLimits:
        forced = self._counts["forced_identity"]
        counts = dict(
            max_draws=self._limits.max_draws - other["draws"] - forced,
            max_values=self._limits.max_values - other["values"],
            max_parameters=self._limits.max_parameters
            - other["parameters"]
            - len(self._events)
            + forced
            - 1,
        )
        if min(counts.values()) <= 0:
            raise ValueError("joint preference limit exceeded")
        return PreferenceLimits(**counts)

    def numerical(self, role: str, request: NumericalParameter) -> NumericalDraw:
        event = self._event(role, "numerical")
        if type(request) is not NumericalParameter:
            raise ValueError("joint numerical sampling requires a frozen request")
        mode = self._mode if request.mode == "meta" else request.mode
        key = (request.name, mode)
        check_numerical_contract(request, mode, self._global_policy, self._contracts.get(key))
        rng = copy.deepcopy(self._rng)
        sampler = self._numerical.fork(rng, limits=self._remaining(self._categorical.admission))
        draw = sampler.numerical(request)
        self._numerical = sampler.fork(self._rng, limits=self._limits)
        self._accept(event, rng)
        self._contracts[key] = request
        return draw

    def forced_identity(self, role: str, *, reason: str) -> JointDecision:
        event = self._event(role, "forced_identity", reason)
        numerical, categorical = self._numerical.admission, self._categorical.admission
        forced = self._counts["forced_identity"]
        self._limits.admit(
            draws=numerical["draws"] + categorical["draws"] + forced + 1,
            values=numerical["values"] + categorical["values"],
            parameters=numerical["parameters"]
            + categorical["parameters"]
            + len(self._events)
            - forced,
        )
        self._accept(event)
        return event

    def categorical(self, role: str, request: CategoricalParameter) -> CategoricalDraw:
        event = self._event(role, "categorical")
        rng = copy.deepcopy(self._rng)
        sampler = self._categorical.fork(rng, limits=self._remaining(self._numerical.admission))
        draw = sampler.categorical(request)
        self._categorical = sampler.fork(self._rng, limits=self._limits)
        self._accept(event, rng)
        return draw

    def freeze(self) -> JointSamplingRecord:
        self._available()
        return JointSamplingRecord(
            self._numerical.freeze(),
            self._categorical.freeze(),
            tuple(self._events),
            global_policy=self._global_policy,
            limits=self._limits,
        )
