from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink
import numpy as np

_HERE = Path(__file__).parent
_XML = _HERE / "unitree_g1" / "scene_no_table.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)
    feet = ["right_foot", "left_foot"]
    hands = ["right_palm", "left_palm"]

    tasks = [
        pelvis_orientation_task := mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        torso_orientation_task := mink.FrameTask(
            frame_name="torso_link",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model, cost=1e-1),
        com_task := mink.ComTask(cost=10.0),
    ]

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=10.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        feet_tasks.append(task)
    tasks.extend(feet_tasks)

    hand_tasks = []
    for hand in hands:
        task = mink.FrameTask(
            frame_name=hand,
            frame_type="site",
            position_cost=5.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        hand_tasks.append(task)
    tasks.extend(hand_tasks)

    # Enable collision avoidance between the following geoms.
    # left hand - table, right hand - table
    # left hand - left thigh, right hand - right thigh
    collision_pairs = [
        # (["left_hand_collision", "right_hand_collision"], ["table"]),
        (["left_hand_collision"], ["left_thigh"]),
        (["right_hand_collision"], ["right_thigh"]),
    ]
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,  # type: ignore
        minimum_distance_from_collisions=0.005,
        collision_detection_distance=0.15,
    )

    limits = [
        mink.ConfigurationLimit(model),
        collision_avoidance_limit,
    ]

    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
    hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in hands]

    model = configuration.model
    data = configuration.data
    solver = "daqp"

    class WaypointTrajectory:
        def __init__(self, poses, pos_tol=0.03, rot_tol=0.5, stable_steps=10, relative_robot_pose=None,
        curr_robot_pose=None):
            # poses: list of [x,y,z, qw,qx,qy,qz]
            self.waypoints = []
            for p in poses:
                pos = p[:3]
                quat = p[3:]
                se3 = mink.SE3.from_rotation_and_translation(
                    rotation=mink.SO3(wxyz=np.array(quat, dtype=float)),
                    translation=np.array(pos, dtype=float),
                )
                self.waypoints.append(se3)
            self.pos_tol = float(pos_tol)
            self.rot_tol = float(rot_tol)
            self.stable_steps = int(stable_steps)
            self.index = 0
            self._stable_count = 0
            self.relative_poses = []
            self.start_pose = curr_robot_pose
            # Preprocess trajectory if robot_pose is provided
            if relative_robot_pose is not None:
                relative_robot_pose = mink.SE3.from_rotation_and_translation(
                    rotation=mink.SO3(wxyz=np.array(relative_robot_pose[3:], dtype=float)),
                    translation=np.array(relative_robot_pose[:3], dtype=float)
                )
                self._preprocess_traj(self.waypoints, relative_robot_pose)

        def current(self):
            """Return the current waypoint (absolute pose)."""
            return self.waypoints[self.index]

        def get_current_target(self, robot_reference_pose):
            """Get the target pose for the current waypoint relative to the robot's current pose.
            
            Args:
                robot_reference_pose: Current SE3 pose of the robot's reference frame
                                      (same frame used in preprocessing)
                
            Returns:
                SE3 target pose in world frame
            """
            if self.relative_poses:
                # Apply current robot reference frame pose to the relative pose
                return self.start_pose @ self.relative_poses[self.index]
            else:
                # Fallback to absolute pose if preprocessing hasn't been done
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

        def _preprocess_traj(self, poses, robot_pose):
            """Compute relative transforms from robot_pose to each pose in poses.
            
            Args:
                poses: List of SE3 poses
                robot_pose: Initial SE3 pose of the robot
            """
            self.relative_poses = []
            for pose in poses:
                # Compute transform from robot frame to target pose
                relative_pose = robot_pose.inverse() @ pose
                self.relative_poses.append(relative_pose)



    # Example trajectory object (user-provided style)
    TRAJ = {
        "poses": [
            [-0.08, 0.1, 0.845, 0.0000, 0.7071, -0.7071, 0.0000],
            [-0.15, 0.1, 0.80, 0.0000, 0.7071, -0.7071, 0.0000],
            [0.15, 0.1, 0.80, 0.0000, 0.7071, -0.7071, 0.0000],
            [-0.20, 0.025, 0.6409, 0.7068956888241477, 0.707317701654041, 0.00018099998471816394, 0.00034826381457648224],
        ],
        "pos_tol": 0.01,
        "rot_tol": 10,
        "stable_steps": 10,
        "base_pose": [0.0, 0.5, 0.755, 0.7071068, 0.0, 0.0, -0.7071068],
    }
    DEBUG_LOG_EVERY = 25  # steps between log lines (set to 1 for every iteration)
    SIM_FREQUENCY = 100.0  # viewer/IK loop rate (Hz); lower = slower overall
    IK_VELOCITY_SCALE = 0.01  # fraction of solved joint velocity to apply (0-1)

    def _quat_angle_rad(q1, q2):
        q1 = np.asarray(q1, dtype=float) / max(np.linalg.norm(q1), 1e-8)
        q2 = np.asarray(q2, dtype=float) / max(np.linalg.norm(q2), 1e-8)
        return float(2.0 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)))

    def _fmt_xyz(v):
        return f"[{v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}]"

    # NOTE: base_pose (if present) is applied after we load the initial keyframe
    # so it is not overwritten. See below where it's applied after the keyframe.

    # with mujoco.viewer.launch_passive(
    #     model=model, data=data, show_left_ui=False, show_right_ui=False
    # ) as viewer:
    #     mujoco.mjv_defaultFreeCamera(model, viewer.cam)

    #     # Initialize to the home keyframe.
    #     configuration.update_from_keyframe("teleop")
    #     posture_task.set_target_from_configuration(configuration)
    #     pelvis_orientation_task.set_target_from_configuration(configuration)
    #     torso_orientation_task.set_target_from_configuration(configuration)
    #     # Apply optional base pose from TRAJ after keyframe to avoid overwrite.
    #     # if "base_pose" in TRAJ:
    #     #     bp = TRAJ["base_pose"]
    #     #     base_pos = np.array(bp[:3], dtype=float)
    #     #     base_quat = np.array(bp[3:], dtype=float)
    #     #     jb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    #     #     if jb != -1:
    #     #         qposadr = int(model.jnt_qposadr[jb])
    #     #         configuration.data.qpos[qposadr : qposadr + 7] = np.concatenate([base_quat, base_pos])
    #     #         configuration.update()
    #     # Initialize mocap bodies at their respective sites.
    #     for hand, foot in zip(hands, feet):
    #         mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
    #         mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
    #     data.mocap_pos[com_mid] = data.subtree_com[1]

    #     rate = RateLimiter(frequency=200.0, warn=False)


    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        # Initialize to the home keyframe.
        configuration.update_from_keyframe("teleop")
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)
        torso_orientation_task.set_target_from_configuration(configuration)
        # Apply optional base pose from TRAJ after keyframe to avoid overwrite.
        # if "base_pose" in TRAJ:
        #     bp = TRAJ["base_pose"]
        #     base_pos = np.array(bp[:3], dtype=float)
        #     base_quat = np.array(bp[3:], dtype=float)
        #     jb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        #     if jb != -1:
        #         qposadr = int(model.jnt_qposadr[jb])
        #         configuration.data.qpos[qposadr : qposadr + 7] = np.concatenate([base_quat, base_pos])
        #         configuration.update()
        # Initialize mocap bodies at their respective sites.
        for hand, foot in zip(hands, feet):
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
        data.mocap_pos[com_mid] = data.subtree_com[1]

        robot_ref_pose = configuration.get_transform_frame_to_world("pelvis", "body")

        rate = RateLimiter(frequency=SIM_FREQUENCY, warn=False)
        step = 0
        traj = WaypointTrajectory(
            TRAJ["poses"], pos_tol=TRAJ["pos_tol"], rot_tol=TRAJ["rot_tol"], stable_steps=TRAJ["stable_steps"],
            relative_robot_pose=TRAJ["base_pose"] if "base_pose" in TRAJ else None,
            curr_robot_pose=robot_ref_pose
        )
        if traj.relative_poses:
            print("Waypoint world targets (from relative transform):")
            for i, rel in enumerate(traj.relative_poses):
                world = robot_ref_pose @ rel
                print(f"  [{i}] {world.translation()}")
            print(f"Current right_palm: {configuration.get_transform_frame_to_world('right_palm', 'site').translation()}")

        while viewer.is_running():
            step += 1

            # Update COM target.
            com_task.set_target(data.mocap_pos[com_mid])

            # Update feet targets.
            for i, foot_task in enumerate(feet_tasks):
                foot_task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))

            # Set right hand target from trajectory.
            # Get the robot's reference frame pose (pelvis)
            robot_ref_pose = configuration.get_transform_frame_to_world("pelvis", "body")
            right_hand_target = traj.get_current_target(robot_ref_pose)
            hand_tasks[0].set_target(right_hand_target)

            # Keep left hand unchanged/mouse-controlled.
            hand_tasks[1].set_target(mink.SE3.from_mocap_id(data, hands_mid[1]))

            # Solve IK.
            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, damping=1e-1, limits=limits
            )
            vel *= IK_VELOCITY_SCALE

            # Apply the IK velocity to the robot configuration.
            configuration.integrate_inplace(vel, rate.dt)

            # Check whether the hand is within tolerances of the current waypoint
            # and advance if it has been stable for the required number of steps.
            err = hand_tasks[0].compute_error(configuration)
            pos_err = np.linalg.norm(err[:3])
            rot_err = np.linalg.norm(err[3:])
            hand_pose = configuration.get_transform_frame_to_world("right_palm", "site")
            hand_pos = hand_pose.translation()
            target_pos = right_hand_target.translation()
            world_pos_err = float(np.linalg.norm(hand_pos - target_pos))
            world_rot_err = _quat_angle_rad(
                hand_pose.rotation().wxyz, right_hand_target.rotation().wxyz
            )
            vel_norm = float(np.linalg.norm(vel))
            within_tol = pos_err <= traj.pos_tol and rot_err <= traj.rot_tol

            if step % DEBUG_LOG_EVERY == 0:
                print(
                    f"[step {step:5d} | wp {traj.index}/{len(traj.waypoints) - 1}] "
                    f"task_err pos={pos_err:.4f} rot={rot_err:.4f} "
                    f"(tol {traj.pos_tol}/{traj.rot_tol}) | "
                    f"world_err pos={world_pos_err:.4f} rot={world_rot_err:.4f} | "
                    f"stable {traj._stable_count}/{traj.stable_steps} "
                    f"within_tol={within_tol} | |vel|={vel_norm:.5f}"
                )
                print(f"  hand world:   {_fmt_xyz(hand_pos)}")
                print(f"  target world: {_fmt_xyz(target_pos)}")
                print(f"  delta world:  {_fmt_xyz(hand_pos - target_pos)}")

            if traj.update_if_stable(pos_err, rot_err):
                print(f"Advanced to waypoint {traj.index}")

            # Draw trajectory waypoints in world frame for debugging.
            viewer.user_scn.ngeom = 0
            for i in range(len(traj.waypoints)):
                if traj.relative_poses:
                    waypoint_world = traj.start_pose @ traj.relative_poses[i]
                else:
                    waypoint_world = traj.waypoints[i]
                if i == traj.index:
                    rgba = np.array([0.0, 1.0, 0.0, 0.8], dtype=np.float32)
                else:
                    rgba = np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float32)
                viewer.user_scn.ngeom += 1
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom - 1],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.02, 0.02, 0.02], dtype=np.float64).reshape(3, 1),
                    waypoint_world.translation().reshape(3, 1),
                    np.eye(3, dtype=np.float32).flatten().reshape(9, 1),
                    rgba.reshape(4, 1),
                )

            viewer.sync()
            rate.sleep()