"""Backend-agnostic task/limit reimplementations for the ManiSkill backend.

mink's :class:`PostureTask` and :class:`ConfigurationLimit` take a
``mujoco.MjModel`` and call ``mujoco.mj_differentiatePos``. For a fixed-base,
revolute-only robot the manifold difference is plain subtraction, so these
numpy reimplementations work against a :class:`ManiSkillConfiguration` instead.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..exceptions import (
    InvalidTarget,
    LimitDefinitionError,
    TargetNotSet,
    TaskDefinitionError,
)
from ..limits.limit import Constraint, Limit
from ..tasks.task import Task
from .configuration import ManiSkillConfiguration


class ManiSkillPostureTask(Task):
    """Regulate joint angles towards a target posture (revolute-only)."""

    target_q: np.ndarray | None

    def __init__(
        self,
        configuration: ManiSkillConfiguration,
        cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        super().__init__(
            cost=np.zeros((configuration.nv,)),
            gain=gain,
            lm_damping=lm_damping,
        )
        self.target_q = None
        self.k = configuration.nv
        self.nq = configuration.nq
        self.set_cost(cost)

    def set_cost(self, cost: npt.ArrayLike) -> None:
        cost = np.atleast_1d(cost)
        if cost.ndim != 1 or cost.shape[0] not in (1, self.k):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} cost must be a vector of shape (1,) "
                f"or ({self.k},). Got {cost.shape}"
            )
        if not np.all(cost >= 0.0):
            raise TaskDefinitionError(f"{self.__class__.__name__} cost should be >= 0")
        self.cost[: self.k] = cost

    def set_target(self, target_q: npt.ArrayLike) -> None:
        target_q = np.atleast_1d(target_q)
        if target_q.ndim != 1 or target_q.shape[0] != self.nq:
            raise InvalidTarget(
                f"Expected target posture of shape ({self.nq},) but got "
                f"{target_q.shape}"
            )
        self.target_q = target_q.copy()

    def set_target_from_configuration(
        self, configuration: ManiSkillConfiguration
    ) -> None:
        self.set_target(configuration.q)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Posture error :math:`e(q) = q \ominus q^* = q - q^*` (revolute)."""
        if self.target_q is None:
            raise TargetNotSet(self.__class__.__name__)
        return configuration.q - self.target_q

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """Posture task Jacobian :math:`J(q) = I_{n_v}`."""
        return np.eye(configuration.nv)


class ManiSkillConfigurationLimit(Limit):
    """Joint position limits as QP inequalities (revolute-only)."""

    def __init__(
        self,
        configuration: ManiSkillConfiguration,
        gain: float = 0.95,
        min_distance_from_limits: float = 0.0,
    ):
        if not 0.0 < gain <= 1.0:
            raise LimitDefinitionError(
                f"{self.__class__.__name__} gain must be in the range (0, 1]"
            )

        self.nv = configuration.nv
        lower = configuration.lower.copy()
        upper = configuration.upper.copy()
        limited = np.isfinite(lower) & np.isfinite(upper)
        lower[limited] += min_distance_from_limits
        upper[limited] -= min_distance_from_limits

        self.lower = lower
        self.upper = upper
        self.indices = np.where(limited)[0]
        self.projection_matrix = (
            np.eye(self.nv)[self.indices] if self.indices.size else None
        )
        self.gain = gain

    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        del dt  # Unused.
        if self.projection_matrix is None:
            return Constraint()
        q = configuration.q
        delta_q_max = self.upper - q
        delta_q_min = q - self.lower
        p_max = self.gain * delta_q_max[self.indices]
        p_min = self.gain * delta_q_min[self.indices]
        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        h = np.hstack([p_max, p_min])
        return Constraint(G=G, h=h)
