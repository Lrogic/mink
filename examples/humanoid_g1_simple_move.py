from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "unitree_g1" / "scene_table.xml"


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
            position_cost=2.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        hand_tasks.append(task)
    tasks.extend(hand_tasks)

    # Enable collision avoidance between the following geoms.
    # left hand - table, right hand - table
    # left hand - left thigh, right hand - right thigh
    collision_pairs = [
        (["left_hand_collision", "right_hand_collision"], ["table"]),
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

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("teleop")
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)
        torso_orientation_task.set_target_from_configuration(configuration)
        # Initialize mocap bodies at their respective sites.
        for hand, foot in zip(hands, feet):
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
        data.mocap_pos[com_mid] = data.subtree_com[1]

        rate = RateLimiter(frequency=200.0, warn=False)
        # Before the viewer loop
        right_start_pose = mink.SE3.from_mocap_id(data, hands_mid[0])
        right_start_pos = right_start_pose.translation().copy()
        right_start_rot = right_start_pose.rotation()

        move_speed = 0.10   # meters per second
        max_up = 0.35       # meters
        elapsed = 0.0

        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            while viewer.is_running():
                elapsed += rate.dt

                # Update COM target.
                com_task.set_target(data.mocap_pos[com_mid])

                # Update feet targets.
                for i, foot_task in enumerate(feet_tasks):
                    foot_task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))

                # Move right hand target upward over time.
                up_amount = min(move_speed * elapsed, max_up)

                right_target_pos = right_start_pos.copy()
                right_target_pos[2] += up_amount

                right_hand_target = mink.SE3.from_rotation_and_translation(
                    rotation=right_start_rot,
                    translation=right_target_pos,
                )

                hand_tasks[0].set_target(right_hand_target)

                # Keep left hand unchanged/mouse-controlled.
                hand_tasks[1].set_target(mink.SE3.from_mocap_id(data, hands_mid[1]))

                # Solve IK.
                vel = mink.solve_ik(
                    configuration, tasks, rate.dt, solver, damping=1e-1, limits=limits
                )

                # Apply the IK velocity to the robot configuration.
                configuration.integrate_inplace(vel, rate.dt)

                viewer.sync()
                rate.sleep()