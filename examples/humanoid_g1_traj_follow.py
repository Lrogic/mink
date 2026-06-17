from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink
import numpy as np

_HERE = Path(__file__).parent
# Set True to weld pelvis at ManiSkill base_pose (no freejoint); uses absolute world waypoints.
FIXED_BASE_TEST = True
# Step MuJoCo physics (gravity + position actuators) instead of kinematic IK only.
# Currently supported with FIXED_BASE_TEST=True only.
DYNAMIC_MODE = True
_XML = _HERE / "unitree_g1" / (
    "scene_no_table_fixed_base.xml" if FIXED_BASE_TEST else "scene_no_table.xml"
)

LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]


def _set_position_actuator_targets(
    model: mujoco.MjModel,
    configuration: mink.Configuration,
    data: mujoco.MjData,
    leg_qpos_hold: dict[str, float] | None = None,
) -> None:
    """Map IK joint targets to position actuator controls."""
    for i in range(model.nu):
        joint_id = int(model.actuator_trnid[i, 0])
        qadr = int(model.jnt_qposadr[joint_id])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if leg_qpos_hold is not None and joint_name in leg_qpos_hold:
            data.ctrl[i] = leg_qpos_hold[joint_name]
        else:
            data.ctrl[i] = configuration.data.qpos[qadr]


def _frame_camera_on_robot(cam, pelvis_pos: np.ndarray, pelvis_rot_matrix: np.ndarray) -> None:
    """Point the free camera at the pelvis, placed behind the robot's facing direction."""
    lookat = pelvis_pos + np.array([0.0, 0.0, 0.25])
    cam.lookat[:] = lookat
    cam.distance = 2.2
    cam.elevation = -15.0
    # Pelvis +X is forward; place the camera on the horizontal opposite side.
    forward = pelvis_rot_matrix[:, 0].copy()
    forward[2] = 0.0
    norm = np.linalg.norm(forward)
    if norm > 1e-6:
        back = -forward / norm
        cam.azimuth = float(np.degrees(np.arctan2(back[1], back[0])))


if __name__ == "__main__":
    if DYNAMIC_MODE and not FIXED_BASE_TEST:
        raise ValueError("DYNAMIC_MODE requires FIXED_BASE_TEST=True for now.")

    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)
    feet = ["right_foot", "left_foot"]
    hands = ["right_palm", "left_palm"]

    tasks = []
    if not FIXED_BASE_TEST:
        tasks.append(
            pelvis_orientation_task := mink.FrameTask(
                frame_name="pelvis",
                frame_type="body",
                position_cost=0.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            )
        )
    tasks.append(
        torso_orientation_task := mink.FrameTask(
            frame_name="torso_link",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
    )
    tasks.append(posture_task := mink.PostureTask(model, cost=1e-1))
    if not FIXED_BASE_TEST:
        tasks.append(com_task := mink.ComTask(cost=10.0))

    feet_tasks = []
    if not FIXED_BASE_TEST:
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
    hand_position_cost = 10.0 if FIXED_BASE_TEST else 10.0
    for hand in hands:
        task = mink.FrameTask(
            frame_name=hand,
            frame_type="site",
            position_cost=hand_position_cost,
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

    ik_constraints = []
    if FIXED_BASE_TEST:
        leg_dof_indices = [
            model.jnt_dofadr[model.joint(joint_name).id] for joint_name in LEG_JOINT_NAMES
        ]
        ik_constraints.append(
            mink.DofFreezingTask(model, dof_indices=leg_dof_indices)
        )

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



    # Same waypoint list in both modes; base_pose converts ManiSkill world poses to
    # robot-relative offsets, then applies them at the current/fixed pelvis pose.
    TRAJ = {
        "poses": [
        [
            -0.2,
            0.025,
            0.6409339256541683,
            0.7068956888241477,
            0.707317701654041,
            0.00018099998471816394,
            0.00034826381457648224
        ],
        [
            -0.184047494794631,
            0.029840778557341213,
            0.6796916034950917,
            0.7081192135306263,
            0.7060770727253313,
            0.00228507555166261,
            -0.004138264143051948
        ],
        [
            -0.168094989589262,
            0.034681557114682425,
            0.7184492813360152,
            0.7093120321707889,
            0.7048281929606509,
            0.00440481286469608,
            -0.008617252995961779
        ],
        [
            -0.152142484383893,
            0.039522335672023644,
            0.7572069591769387,
            0.710474110674306,
            0.7035712159246243,
            0.006540172308655456,
            -0.013088480816079127
        ],
        [
            -0.13618997917852402,
            0.044363114229364856,
            0.7959646370178621,
            0.7116054166512511,
            0.702306295335484,
            0.008691112908214782,
            -0.01755172587422396
        ],
        [
            -0.12023747397315503,
            0.04920389278670606,
            0.8347223148587856,
            0.7127059193926731,
            0.7010335850534819,
            0.010857592328942189,
            -0.02200676665313765
        ],
        [
            -0.10428496876778601,
            0.05404467134404728,
            0.8734799926997091,
            0.7137755898710701,
            0.6997532390692954,
            0.01303956687845227,
            -0.026453381860508614
        ],
        [
            -0.07152128060869672,
            0.050649520466326314,
            0.8808452602532724,
            0.7152086322308332,
            0.6982790105170058,
            0.019954818584649384,
            -0.022019106910724305
        ],
        [
            -0.03205949217680086,
            0.043972918107905254,
            0.8757028112179035,
            0.7167156252295368,
            0.6966327277423,
            0.02872360172373217,
            -0.014018198677907892
        ],
        [
            0.0074022962550949895,
            0.037296315749484193,
            0.8705603621825346,
            0.7181302304234835,
            0.6948741475792319,
            0.03745793774747947,
            -0.005982815194902531
        ],
        [
            0.04686408468699085,
            0.030619713391063137,
            0.8654179131471657,
            0.7194518148186637,
            0.6930038422153517,
            0.046156361033170536,
            0.0020859446461802167
        ],
        [
            0.08632587311888673,
            0.023943111032642073,
            0.8602754641117968,
            0.7206797588404965,
            0.6910224033035107,
            0.054817414607564834,
            0.010186971384389579
        ],
        [
            0.12578766155078264,
            0.017266508674221005,
            0.8551330150764279,
            0.7218134565014327,
            0.688930441837256,
            0.06343965040945945,
            0.01831914515753878
        ],
        [
            0.16524944998267846,
            0.010589906315799952,
            0.8499905660410589,
            0.7228523155664305,
            0.6867285880226294,
            0.07202162955049983,
            0.026481335872383844
        ],
        [
            0.17766279309142302,
            0.006790837768152886,
            0.8274478045173825,
            0.7217892441907463,
            0.6881684418409949,
            0.06862063379373257,
            0.027123628858154388
        ],
        [
            0.16713198283221042,
            0.00543267021452231,
            0.7901450287447398,
            0.7189495822277903,
            0.692548501379255,
            0.05502313642880732,
            0.021459867412698806
        ],
        [
            0.15660117257299777,
            0.004074502660891731,
            0.7528422529720968,
            0.7160447140068642,
            0.6966465028645852,
            0.04135623092252526,
            0.01591476525917259
        ],
        [
            0.14607036231378517,
            0.0027163351072611525,
            0.715539477199454,
            0.7130782395015844,
            0.700461422461218,
            0.027626003163011503,
            0.01048922981754308
        ],
        [
            0.13553955205457258,
            0.0013581675536305767,
            0.6782367014268111,
            0.7100537727290245,
            0.7039923718656851,
            0.01383855476926566,
            0.005184070804310106
        ],
        [
            0.12500874179535995,
            0.0,
            0.6409339256541683,
            0.7069749392341962,
            0.7072385985611961,
            0.0,
            0.0
        ]
    ],
        "pos_tol": 0.03 if FIXED_BASE_TEST else 0.03,
        "rot_tol": 10,
        "stable_steps": 10,
        "base_pose": [0.0, 0.5, 0.755, 0.7071068, 0.0, 0.0, -0.7071068],
    }
    DEBUG_LOG_EVERY = 25  # steps between log lines (set to 1 for every iteration)
    SIM_FREQUENCY = 100.0  # viewer/IK loop rate (Hz); lower = slower overall
    IK_VELOCITY_SCALE = 0.1 if FIXED_BASE_TEST else 0.1

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
        # Initialize to the home keyframe.
        configuration.update_from_keyframe("teleop")
        configuration.update()
        pelvis_pose = configuration.get_transform_frame_to_world("pelvis", "body")
        _frame_camera_on_robot(
            viewer.cam,
            pelvis_pose.translation(),
            pelvis_pose.rotation().as_matrix(),
        )
        posture_task.set_target_from_configuration(configuration)
        if not FIXED_BASE_TEST:
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

        leg_qpos_hold = {
            name: float(
                configuration.data.qpos[model.jnt_qposadr[model.joint(name).id]]
            )
            for name in LEG_JOINT_NAMES
        }
        print(f"leg_qpos_hold: {leg_qpos_hold}")
        print(f"configuration.data.qpos: {configuration.data.qpos}")
        print(f"model.jnt_qposadr: {model.jnt_qposadr}")
        print(f"length of configuration.data.qpos: {len(configuration.data.qpos)}")
        print(f"length of model.jnt_qposadr: {len(model.jnt_qposadr)}")
        print(f"length of LEG_JOINT_NAMES: {len(LEG_JOINT_NAMES)}")
        print(f"length of leg_qpos_hold: {len(leg_qpos_hold)}")
        # print(f"length of model.joint(name).id: {len(model.joint("left_hip_pitch_joint").id)}")
        # print(f"length of model.joint(name).id: {len(model.joint("left_hip_pitch_joint").id)}")
        if DYNAMIC_MODE:
            _set_position_actuator_targets(
                model, configuration, data, leg_qpos_hold=leg_qpos_hold
            )
            mujoco.mj_forward(model, data)

        robot_ref_pose = configuration.get_transform_frame_to_world("pelvis", "body")

        rate = RateLimiter(frequency=SIM_FREQUENCY, warn=False)
        step = 0
        traj = WaypointTrajectory(
            TRAJ["poses"], pos_tol=TRAJ["pos_tol"], rot_tol=TRAJ["rot_tol"], stable_steps=TRAJ["stable_steps"],
            relative_robot_pose=TRAJ["base_pose"],
            curr_robot_pose=robot_ref_pose,
        )
        if DYNAMIC_MODE:
            print(
                "DYNAMIC_MODE: closed-loop IK + mj_step "
                f"({SIM_FREQUENCY:.0f} Hz, gravity on)."
            )
        if FIXED_BASE_TEST:
            print("FIXED_BASE_TEST: pelvis welded, leg DoFs frozen, absolute waypoints.")
            print(f"Pelvis world: {robot_ref_pose.translation()}")
            for i, wp in enumerate(traj.waypoints):
                print(f"  WP[{i}] target: {wp.translation()}")
            print(
                "Initial right_palm: "
                f"{configuration.get_transform_frame_to_world('right_palm', 'site').translation()}"
            )
        elif traj.relative_poses:
            print("Waypoint world targets (from relative transform):")
            for i, rel in enumerate(traj.relative_poses):
                world = robot_ref_pose @ rel
                print(f"  [{i}] {world.translation()}")
            print(f"Current right_palm: {configuration.get_transform_frame_to_world('right_palm', 'site').translation()}")

        while viewer.is_running():
            step += 1

            if DYNAMIC_MODE:
                configuration.update(data.qpos)

            # Update COM target.
            if not FIXED_BASE_TEST:
                com_task.set_target(data.mocap_pos[com_mid])

            # Update feet targets.
            if not FIXED_BASE_TEST:
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
                configuration,
                tasks,
                rate.dt,
                solver,
                damping=1e-1,
                limits=limits,
                constraints=ik_constraints or None,
            )
            vel *= IK_VELOCITY_SCALE

            # Apply the IK velocity to the robot configuration.
            configuration.integrate_inplace(vel, rate.dt) 
            if DYNAMIC_MODE:
                _set_position_actuator_targets(
                    model, configuration, data, leg_qpos_hold=leg_qpos_hold
                )
                mujoco.mj_step(model, data)
                configuration.update(data.qpos)

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