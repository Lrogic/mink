"""SAPIEN/ManiSkill-flavored configuration backed by ``pytorch_kinematics``.

:class:`ManiSkillConfiguration` mirrors the subset of
:class:`mink.configuration.Configuration` that the G1 hand-tracking demo needs,
but computes forward kinematics and frame Jacobians from a URDF using
``pytorch_kinematics`` instead of MuJoCo. The robot is assumed to be a
fixed-base, revolute-only articulation so that ``nq == nv`` and velocity
integration is plain addition.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytorch_kinematics as pk
import torch

from ..exceptions import MinkError
from ..lie import SE3, SO3

# Frame types accepted for parity with mink. Every supported frame resolves to a
# URDF link (the demo's "site" palm targets are real links, e.g. right_tcp_link).
SUPPORTED_FRAMES = ("body", "geom", "site")


class ManiSkillConfiguration:
    """Forward kinematics and Jacobians for a fixed-base revolute robot.

    Args:
        urdf_path: Path to the robot URDF.
        q: Optional initial configuration of shape ``(nv,)``. Defaults to zeros.
        base_pose: Optional world pose of the URDF root link. Frame transforms
            returned by :meth:`get_transform_frame_to_world` are expressed in
            this world frame. Defaults to identity.
    """

    def __init__(
        self,
        urdf_path: str,
        q: np.ndarray | None = None,
        base_pose: SE3 | None = None,
    ):
        with open(urdf_path, "rb") as f:
            urdf_bytes = f.read()

        self.chain = pk.build_chain_from_urdf(urdf_bytes).to(dtype=torch.float64)
        self.joint_names: list[str] = self.chain.get_joint_parameter_names()
        self._nv = len(self.joint_names)
        self._col_of: dict[str, int] = {n: i for i, n in enumerate(self.joint_names)}
        self._serial_chains: dict[str, pk.SerialChain] = {}
        self._urdf_bytes = urdf_bytes
        self._eye_nv = np.eye(self._nv)

        lower, upper = self._parse_joint_limits(urdf_bytes)
        self.lower = lower
        self.upper = upper

        self.base_pose = base_pose if base_pose is not None else SE3.identity()

        self._q = (
            np.zeros(self._nv)
            if q is None
            else np.asarray(q, dtype=np.float64).copy()
        )
        self._fk: dict = {}
        self.update()

    def _parse_joint_limits(self, urdf_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
        """Read per-joint [lower, upper] limits from the URDF in joint order."""
        root = ET.fromstring(urdf_bytes)
        limits: dict[str, tuple[float, float]] = {}
        for joint in root.findall("joint"):
            name = joint.get("name")
            limit = joint.find("limit")
            if name is None or limit is None:
                continue
            lo = limit.get("lower")
            hi = limit.get("upper")
            if lo is not None and hi is not None:
                limits[name] = (float(lo), float(hi))
        lower = np.full(self._nv, -np.inf)
        upper = np.full(self._nv, np.inf)
        for i, name in enumerate(self.joint_names):
            if name in limits:
                lower[i], upper[i] = limits[name]
        return lower, upper

    def update(self, q: np.ndarray | None = None) -> None:
        """Run forward kinematics, optionally overriding the configuration."""
        if q is not None:
            self._q = np.asarray(q, dtype=np.float64).copy()
        th = torch.as_tensor(self._q, dtype=torch.float64)
        self._fk = self.chain.forward_kinematics(th)

    def _resolve_frame(self, frame_name: str, frame_type: str) -> str:
        if frame_type not in SUPPORTED_FRAMES:
            raise MinkError(
                f"Unsupported frame type '{frame_type}'. "
                f"Supported types: {SUPPORTED_FRAMES}"
            )
        if frame_name not in self._fk:
            raise MinkError(
                f"Frame '{frame_name}' (type '{frame_type}') is not a link in the "
                f"URDF. Available links: {sorted(self._fk.keys())}"
            )
        return frame_name

    def _link_pose_in_base(self, link_name: str) -> SE3:
        mat = self._fk[link_name].get_matrix()[0].cpu().numpy()
        return SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(mat[:3, :3]),
            translation=mat[:3, 3],
        )

    def _get_transform_frame_to_world_wxyz_xyz(
        self, frame_name: str, frame_type: str
    ) -> np.ndarray:
        link = self._resolve_frame(frame_name, frame_type)
        world = self.base_pose @ self._link_pose_in_base(link)
        return world.wxyz_xyz

    def get_transform_frame_to_world(self, frame_name: str, frame_type: str) -> SE3:
        """Pose of a frame in the world frame at the current configuration."""
        return SE3(
            wxyz_xyz=self._get_transform_frame_to_world_wxyz_xyz(frame_name, frame_type)
        )

    def get_transform(
        self,
        source_name: str,
        source_type: str,
        dest_name: str,
        dest_type: str,
    ) -> SE3:
        """Pose of ``source`` expressed in ``dest`` at the current configuration."""
        source = self.get_transform_frame_to_world(source_name, source_type)
        dest = self.get_transform_frame_to_world(dest_name, dest_type)
        return dest.inverse() @ source

    def _serial_chain(self, link_name: str) -> pk.SerialChain:
        if link_name not in self._serial_chains:
            self._serial_chains[link_name] = pk.SerialChain(
                self.chain, link_name
            ).to(dtype=torch.float64)
        return self._serial_chains[link_name]

    def get_frame_jacobian(self, frame_name: str, frame_type: str) -> np.ndarray:
        r"""Body Jacobian :math:`{}_B J_{WB}` of a frame, shape ``(6, nv)``.

        ``pytorch_kinematics`` returns a base-frame geometric Jacobian (rows are
        ``[linear; angular]``) over the joints on the kinematic path to the
        frame. We rotate it into the frame's local (body) frame to match mink's
        convention, and scatter the path columns into the full ``nv`` width
        (off-path joints do not move the frame, so their columns are zero).
        """
        link = self._resolve_frame(frame_name, frame_type)
        serial = self._serial_chain(link)
        serial_names = serial.get_joint_parameter_names()
        th = torch.as_tensor(
            [self._q[self._col_of[n]] for n in serial_names], dtype=torch.float64
        )
        jac_base = serial.jacobian(th)[0].cpu().numpy()  # (6, len(serial_names))

        # Rotate the base-frame Jacobian into the body frame.
        R_base_link = self._fk[link].get_matrix()[0].cpu().numpy()[:3, :3]
        R_link_base = R_base_link.T
        lin = R_link_base @ jac_base[:3]
        ang = R_link_base @ jac_base[3:]

        jac = np.zeros((6, self._nv))
        for k, name in enumerate(serial_names):
            col = self._col_of[name]
            jac[:3, col] = lin[:, k]
            jac[3:, col] = ang[:, k]
        return jac

    def integrate(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        """Return the configuration after integrating ``velocity`` for ``dt``."""
        return self._q + np.asarray(velocity, dtype=np.float64) * dt

    def integrate_inplace(self, velocity: np.ndarray, dt: float) -> None:
        """Integrate ``velocity`` for ``dt`` and refresh forward kinematics."""
        self._q = self._q + np.asarray(velocity, dtype=np.float64) * dt
        self.update()

    def check_limits(self, tol: float = 1e-6, safety_break: bool = True) -> None:
        """Check that the current configuration is within joint limits."""
        limited = np.isfinite(self.lower) & np.isfinite(self.upper)
        if not limited.any():
            return
        violations = (self._q < self.lower - tol) | (self._q > self.upper + tol)
        violations &= limited
        if not violations.any():
            return
        if safety_break:
            idx = int(np.argmax(violations))
            raise MinkError(
                f"Joint {idx} ({self.joint_names[idx]}) violates configuration "
                f"limits {self.lower[idx]} <= {self._q[idx]} <= {self.upper[idx]}"
            )

    @property
    def q(self) -> np.ndarray:
        """The current configuration vector."""
        return self._q.copy()

    @property
    def nv(self) -> int:
        """The dimension of the tangent space."""
        return self._nv

    @property
    def nq(self) -> int:
        """The dimension of the configuration space (``== nv`` for revolute)."""
        return self._nv

    @property
    def model(self) -> SimpleNamespace:
        """Lightweight shim exposing ``nv`` for mink's solver internals.

        mink's ``solve_ik`` reads ``configuration.model.nv``; this avoids
        needing a real ``mujoco.MjModel`` for the ManiSkill backend.
        """
        return SimpleNamespace(nv=self._nv)
