"""Parity check: ManiSkillConfiguration (pytorch_kinematics) vs MuJoCo mink.

Validates that the ManiSkill/SAPIEN kinematics backend produces the same frame
poses and body Jacobians as the reference MuJoCo ``mink.Configuration`` for the
fixed-base G1 at the ``teleop`` keyframe. Poses are compared relative to the
``pelvis`` (root) frame so no world-placement bookkeeping is needed. Jacobian
columns are aligned by joint name.

Run with the mink ``src`` on PYTHONPATH and the native extension disabled, e.g.::

    PYTHONPATH=src MINK_DISABLE_NATIVE=1 python examples/check_parity_g1_maniskill.py
"""

from pathlib import Path

import mujoco
import numpy as np

import mink
from mink.maniskill import ManiSkillConfiguration

_HERE = Path(__file__).parent
_MJCF = _HERE / "unitree_g1" / "scene_no_table_fixed_base.xml"
_URDF = _HERE / "unitree_g1" / "g1_29dof_with_hand_rev_1_0.urdf"

# (mujoco_frame_name, mujoco_frame_type, maniskill_link_name)
FRAME_PAIRS = [
    ("torso_link", "body", "torso_link"),
    ("right_palm", "site", "right_tcp_link"),
    ("left_palm", "site", "left_tcp_link"),
]


def _rotation_angle(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle [rad] between two rotation matrices."""
    R = R_a.T @ R_b
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def main() -> None:
    model = mujoco.MjModel.from_xml_path(_MJCF.as_posix())
    cfg_mj = mink.Configuration(model)
    cfg_mj.update_from_keyframe("teleop")

    # Joint value per name from MuJoCo (all hinge -> 1 qpos each).
    q_mj = cfg_mj.q
    name_to_val: dict[str, float] = {}
    for j in range(model.njnt):
        name = model.joint(j).name
        qadr = int(model.jnt_qposadr[j])
        name_to_val[name] = float(q_mj[qadr])

    # DOF order (for Jacobian column alignment).
    mj_dof_names = [""] * model.nv
    for j in range(model.njnt):
        dadr = int(model.jnt_dofadr[j])
        mj_dof_names[dadr] = model.joint(j).name

    cfg_ms = ManiSkillConfiguration(_URDF.as_posix())
    q_ms = np.array([name_to_val[n] for n in cfg_ms.joint_names])
    cfg_ms.update(q_ms)

    pelvis_mj = "pelvis"

    max_pos_err = 0.0
    max_rot_err = 0.0
    max_jac_err = 0.0
    for mj_name, mj_type, ms_link in FRAME_PAIRS:
        # Pose relative to pelvis.
        T_mj = cfg_mj.get_transform(mj_name, mj_type, pelvis_mj, "body")
        T_ms = cfg_ms.get_transform_frame_to_world(ms_link, "body")  # base = identity
        pos_err = float(np.linalg.norm(T_mj.translation() - T_ms.translation()))
        rot_err = _rotation_angle(
            T_mj.rotation().as_matrix(), T_ms.rotation().as_matrix()
        )

        # Body Jacobians, columns aligned to MuJoCo DOF order by joint name.
        J_mj = cfg_mj.get_frame_jacobian(mj_name, mj_type)
        J_ms_raw = cfg_ms.get_frame_jacobian(ms_link, "body")
        J_ms = np.zeros_like(J_mj)
        for i, dof_name in enumerate(mj_dof_names):
            J_ms[:, i] = J_ms_raw[:, cfg_ms._col_of[dof_name]]
        jac_err = float(np.max(np.abs(J_mj - J_ms)))

        max_pos_err = max(max_pos_err, pos_err)
        max_rot_err = max(max_rot_err, rot_err)
        max_jac_err = max(max_jac_err, jac_err)
        print(
            f"{mj_name:>12s} ({mj_type}) vs {ms_link:>16s}: "
            f"pos_err={pos_err:.2e}  rot_err={rot_err:.2e}  jac_err={jac_err:.2e}"
        )

    print(
        f"\nMAX  pos_err={max_pos_err:.2e}  rot_err={max_rot_err:.2e}  "
        f"jac_err={max_jac_err:.2e}"
    )
    pos_tol, rot_tol, jac_tol = 1e-3, 1e-3, 1e-3
    ok = max_pos_err < pos_tol and max_rot_err < rot_tol and max_jac_err < jac_tol
    print("PARITY:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
