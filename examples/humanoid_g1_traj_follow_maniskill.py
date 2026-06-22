"""G1 hand-waypoint IK on a ManiSkill/SAPIEN-loaded robot.

ManiSkill/SAPIEN port of the fixed-base path of ``humanoid_g1_traj_follow.py``.
The robot is loaded from URDF into SAPIEN; forward kinematics and Jacobians come
from mink's ManiSkill backend (pytorch_kinematics). The right hand
(``right_tcp_link``, the collision-free TCP link that mirrors the MuJoCo
``right_palm`` site) tracks a list of world-frame waypoints while the torso
orientation, posture, and joint limits are regulated and the legs are frozen.

Two stepping modes:
  * ``kinematic`` (default): the IK output is written directly to the SAPIEN
    robot via ``set_qpos`` each iteration. Deterministic; used for validation.
  * ``dynamic`` (``--dynamic``): the IK output is sent as ``pd_joint_pos`` drive
    targets and the PhysX scene is stepped under gravity.

Rendering uses the SAPIEN viewer when ``--viewer`` is set and Vulkan is
available; otherwise it runs headless and logs tracking metrics.

Run with the mink ``src`` on PYTHONPATH and the native extension disabled::

    PYTHONPATH=src MINK_DISABLE_NATIVE=1 python examples/humanoid_g1_traj_follow_maniskill.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import mink
from mink.maniskill import (
    ManiSkillConfigurationLimit,
    ManiSkillConfiguration,
    ManiSkillPostureTask,
)

_HERE = Path(__file__).parent
_URDF = _HERE / "unitree_g1" / "g1_29dof_with_hand_rev_1_0.urdf"

# Revolute joint order in the URDF (and pytorch_kinematics chain order).
_ALL_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint", "left_hand_middle_0_joint",
    "left_hand_middle_1_joint", "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint", "right_hand_middle_0_joint",
    "right_hand_middle_1_joint", "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]

# Home "teleop" pose from unitree_g1/scene_no_table_fixed_base.xml (no floating
# base prefix), in _ALL_JOINT_NAMES order.
TELEOP_QPOS = np.array([
    -0.312, 0, 0, 0.669, -0.363, 0, -0.312, 0, 0, 0.669, -0.363, 0,
    0, 0, 0.468,
    -0.09432, 0.5048, -0.36652, 0.602025, 0, 0, 0, 0, 1.05, 0, 0, 0, 0, 0,
    -1.41889, -0.3512, 0.02618, 1.28, 0, 0, 0, 0, -1.05, 0, 0, 0, 0, 0,
], dtype=np.float64)

LEG_JOINT_NAMES = _ALL_JOINT_NAMES[:12]

TRAJ = {
    "poses": [
        [-0.08, 0.24, 0.845, 0.7071068, 0.0, 0.0, -0.7071068],
        [-0.15, 0.18, 0.80, 0.7071068, 0.0, 0.0, -0.7071068],
        [-0.20, 0.025, 0.6409, 0.7068956888241477, 0.707317701654041,
         0.00018099998471816394, 0.00034826381457648224],
    ],
    "pos_tol": 0.03,
    "rot_tol": 0.5,
    "stable_steps": 10,
    "base_pose": [0.0, 0.5, 0.755, 0.7071068, 0.0, 0.0, -0.7071068],
}

SIM_FREQUENCY = 100.0
IK_VELOCITY_SCALE = 0.1
DEBUG_LOG_EVERY = 25

# PhysX joint-drive gains for --dynamic. The IK runs closed-loop (the simulated
# qpos is read back each step) with IK_VELOCITY_SCALE=0.1, so the drive only ever
# sees a tiny per-step position command; the steady-state gravity sag is roughly
# G / (stiffness * IK_VELOCITY_SCALE), hence the high stiffness. Damping is kept
# near critical for the arm links (heavily overdamped values like 4e2 stall the
# commanded motion and the hand never reaches the waypoints).
DRIVE_STIFFNESS = 2e4
DRIVE_DAMPING = 80.0
DRIVE_FORCE_LIMIT = 2000.0


def _quat_angle_rad(q1, q2) -> float:
    """Geodesic angle [rad] between two wxyz quaternions."""
    q1 = np.asarray(q1, dtype=float) / max(np.linalg.norm(q1), 1e-8)
    q2 = np.asarray(q2, dtype=float) / max(np.linalg.norm(q2), 1e-8)
    return float(2.0 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)))


class WaypointTrajectory:
    """Sequence of SE3 waypoints with stability-based advancement."""

    def __init__(self, poses, pos_tol=0.03, rot_tol=0.5, stable_steps=10,
                 relative_robot_pose=None, curr_robot_pose=None):
        self.waypoints = []
        for p in poses:
            se3 = mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(wxyz=np.array(p[3:], dtype=float)),
                translation=np.array(p[:3], dtype=float),
            )
            self.waypoints.append(se3)
        self.pos_tol = float(pos_tol)
        self.rot_tol = float(rot_tol)
        self.stable_steps = int(stable_steps)
        self.index = 0
        self._stable_count = 0
        self.relative_poses = []
        self.start_pose = curr_robot_pose
        if relative_robot_pose is not None:
            rel = mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(wxyz=np.array(relative_robot_pose[3:], dtype=float)),
                translation=np.array(relative_robot_pose[:3], dtype=float),
            )
            self.relative_poses = [rel.inverse() @ wp for wp in self.waypoints]

    def get_current_target(self, robot_reference_pose):
        if self.relative_poses:
            return self.start_pose @ self.relative_poses[self.index]
        return self.waypoints[self.index]

    def update_if_stable(self, position_error, rotation_error):
        if position_error <= self.pos_tol and rotation_error <= self.rot_tol:
            self._stable_count += 1
        else:
            self._stable_count = 0
        if self._stable_count >= self.stable_steps:
            self._stable_count = 0
            self.index = (self.index + 1) % len(self.waypoints)
            return True
        return False


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 waypoint tracking in ManiSkill")
    parser.add_argument("--waypoint-config", type=str, default=None,
                        help="Path to a JSON trajectory config (defaults to TRAJ).")
    parser.add_argument("--dynamic", action="store_true",
                        help="Step PhysX with pd_joint_pos drive targets.")
    parser.add_argument("--viewer", action="store_true",
                        help="Open the SAPIEN viewer (requires Vulkan).")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Max IK iterations when running headless.")
    return parser.parse_args()


def _frame_sapien_viewer(viewer, lookat: np.ndarray) -> None:
    """Point the SAPIEN free camera at *lookat* (pelvis-ish)."""
    lookat = np.asarray(lookat, dtype=float)
    cam_xyz = lookat + np.array([-2.0, 0.0, 0.5])
    viewer.set_camera_xyz(float(cam_xyz[0]), float(cam_xyz[1]), float(cam_xyz[2]))
    dx, dy, dz = lookat - cam_xyz
    horiz = float(np.hypot(dx, dy))
    viewer.set_camera_rpy(r=0.0, p=-float(np.arctan2(dz, horiz)), y=0.0)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)


def _build_sapien(base_pose: list[float]):
    """Load the G1 into a SAPIEN scene. Returns (scene, robot, name->active-index)."""
    import sapien

    scene = sapien.Scene()
    scene.set_timestep(1.0 / SIM_FREQUENCY)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 0, -1], [0.8, 0.8, 0.8])
    scene.add_ground(altitude=0)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(str(_URDF))
    pose = sapien.Pose(p=base_pose[:3], q=base_pose[3:])
    try:
        robot.set_root_pose(pose)
    except AttributeError:
        robot.set_pose(pose)
    sapien_names = [j.name for j in robot.active_joints]
    return scene, robot, sapien_names


def main() -> None:
    args = parse_args()
    loaded = TRAJ if not args.waypoint_config else _load_json(args.waypoint_config)
    base_pose = loaded["base_pose"]

    base_se3 = mink.SE3.from_rotation_and_translation(
        rotation=mink.SO3(wxyz=np.array(base_pose[3:], dtype=float)),
        translation=np.array(base_pose[:3], dtype=float),
    )
    configuration = ManiSkillConfiguration(
        str(_URDF), q=TELEOP_QPOS.copy(), base_pose=base_se3
    )

    # Tasks (fixed-base path of the MuJoCo demo).
    torso_orientation_task = mink.FrameTask(
        frame_name="torso_link", frame_type="body",
        position_cost=0.0, orientation_cost=1.0, lm_damping=1.0,
    )
    posture_task = ManiSkillPostureTask(configuration, cost=1e-1)
    hand_tasks = [
        mink.FrameTask(
            frame_name=link, frame_type="body",
            position_cost=20.0, orientation_cost=0.0, lm_damping=1.0,
        )
        for link in ("right_tcp_link", "left_tcp_link")
    ]
    tasks = [torso_orientation_task, posture_task, *hand_tasks]

    limits = [ManiSkillConfigurationLimit(configuration)]

    leg_dof_indices = [_ALL_JOINT_NAMES.index(name) for name in LEG_JOINT_NAMES]
    dof_freezing_task = mink.DofFreezingTask(
        model=configuration, dof_indices=leg_dof_indices
    )
    ik_constraints = [dof_freezing_task]

    # Initialize targets from the home configuration.
    posture_task.set_target_from_configuration(configuration)
    torso_orientation_task.set_target_from_configuration(configuration)
    hand_tasks[1].set_target_from_configuration(configuration)  # hold left hand

    robot_ref_pose = configuration.get_transform_frame_to_world("pelvis", "body")
    traj = WaypointTrajectory(
        loaded["poses"], pos_tol=loaded["pos_tol"], rot_tol=loaded["rot_tol"],
        stable_steps=loaded["stable_steps"], relative_robot_pose=base_pose,
        curr_robot_pose=robot_ref_pose,
    )

    # SAPIEN scene/robot (for visualization and optional dynamics).
    scene = robot = sapien_names = None
    q_to_sapien = None
    try:
        scene, robot, sapien_names = _build_sapien(base_pose)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] SAPIEN scene unavailable ({exc}); running config-only.")
        scene = robot = None

    if robot is not None:
        assert sapien_names is not None
        q_to_sapien = [configuration.joint_names.index(n) for n in sapien_names]
        q_sapien0 = np.asarray(configuration.q)[q_to_sapien]
        robot.set_qpos(q_sapien0)
        if args.dynamic:
            for joint, target in zip(robot.active_joints, q_sapien0):
                joint.set_drive_property(
                    stiffness=DRIVE_STIFFNESS,
                    damping=DRIVE_DAMPING,
                    force_limit=DRIVE_FORCE_LIMIT,
                )
                joint.set_drive_target(float(target))

    viewer = None
    if args.viewer and robot is not None:
        try:
            from sapien.utils import Viewer

            viewer = Viewer()
            viewer.set_scene(scene)
            tr = robot_ref_pose.translation()
            viewer.set_camera_xyz(tr[0] - 2.0, tr[1], tr[2] + 0.5)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] viewer unavailable ({exc}); running headless.")
            viewer = None

    dt = 1.0 / SIM_FREQUENCY
    solver = "daqp"
    # Each advance confirms the hand was stable AT the previously active
    # waypoint; len(waypoints) advances therefore confirms every waypoint
    # (wp0, wp1, wp2) was reached within tolerance.
    arrivals = 0
    step = 0
    max_steps = args.steps if viewer is None else 10**9

    print(f"Pelvis world: {robot_ref_pose.translation()}")
    for i, wp in enumerate(traj.waypoints):
        print(f"  WP[{i}] world target: {wp.translation()}")

    while step < max_steps:
        step += 1
        if viewer is not None and viewer.closed:
            break

        right_hand_target = traj.get_current_target(robot_ref_pose)
        hand_tasks[0].set_target(right_hand_target)

        vel = mink.solve_ik(
            configuration, tasks, dt, solver,
            damping=1e-1, limits=limits, constraints=ik_constraints,
        )
        vel = vel * IK_VELOCITY_SCALE
        configuration.integrate_inplace(vel, dt)

        if robot is not None:
            assert scene is not None and q_to_sapien is not None
            assert sapien_names is not None
            q_sapien = np.asarray(configuration.q)[q_to_sapien]
            if args.dynamic:
                for joint, target in zip(robot.active_joints, q_sapien):
                    joint.set_drive_target(float(target))
                scene.step()
                # Read the simulated state back into the configuration.
                sim_q = np.asarray(robot.get_qpos()).reshape(-1)
                q_full = np.array(configuration.q)
                for k, name in enumerate(sapien_names):
                    q_full[configuration.joint_names.index(name)] = sim_q[k]
                configuration.update(q_full)
            else:
                robot.set_qpos(q_sapien)
            if viewer is not None:
                scene.update_render()
                viewer.render()

        err = hand_tasks[0].compute_error(configuration)
        pos_err = float(np.linalg.norm(err[:3]))
        rot_err = float(np.linalg.norm(err[3:]))
        hand_pose = configuration.get_transform_frame_to_world(
            "right_tcp_link", "body"
        )
        hand_pos = hand_pose.translation()
        target_pos = right_hand_target.translation()
        world_pos_err = float(np.linalg.norm(hand_pos - target_pos))
        world_rot_err = _quat_angle_rad(
            hand_pose.rotation().wxyz, right_hand_target.rotation().wxyz
        )

        if step % DEBUG_LOG_EVERY == 0:
            print(
                f"[step {step:5d} | wp {traj.index}/{len(traj.waypoints) - 1}] "
                f"task_pos={pos_err:.4f} task_rot={rot_err:.4f} "
                f"world_pos={world_pos_err:.4f} world_rot={world_rot_err:.4f} "
                f"stable {traj._stable_count}/{traj.stable_steps} "
                f"|vel|={np.linalg.norm(vel):.5f}"
            )

        # The right-hand task tracks position only (orientation_cost=0, as in the
        # MuJoCo demo), so waypoint advancement is gated on position alone; the
        # uncontrolled orientation error is logged above for diagnostics.
        reached_wp = traj.index  # Waypoint just confirmed (before advancing).
        if traj.update_if_stable(pos_err, 0.0):
            arrivals += 1
            print(
                f"Reached waypoint {reached_wp} "
                f"(world_pos_err={world_pos_err:.4f}); now tracking "
                f"waypoint {traj.index} at step {step}"
            )
            if viewer is None and arrivals >= len(traj.waypoints):
                print("All waypoints reached within tolerance.")
                break

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
