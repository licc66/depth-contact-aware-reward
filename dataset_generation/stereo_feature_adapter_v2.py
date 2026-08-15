"""Observation-only stereo geometry with an explicit gripper locator.

The v1 adapter estimates the manipulated object and support/goal geometry.  This
v2 adapter keeps that contract and additionally triangulates visible gripper
links from renderer segmentation plus SGBM disparity.  The gripper estimate is
an observable image-derived proxy, not the simulator TCP pose.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def _load_wrapper_module():
    module_name = "reward_wrapper_for_stereo_v2"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = SCRIPT_DIR / "30_maniskill_reward_wrapper_v1.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load wrapper module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


wrapper = _load_wrapper_module()


GRIPPER_GEOMETRY_FEATURES = {
    "gripper_cam_x",
    "gripper_cam_y",
    "gripper_cam_depth_m",
    "gripper_object_3d_dist_m",
    "gripper_goal_3d_dist_m",
    "gripper_object_cam_lateral_error_m",
    "gripper_object_cam_depth_error_m",
    "left_finger_cam_depth_m",
    "right_finger_cam_depth_m",
}


class ManiSkillStereoFeatureAdapterV2(wrapper.ManiSkillStereoFeatureAdapterV1):
    """V1 SGBM geometry plus an image-derived gripper-center proxy."""

    @staticmethod
    def _hand_link(ue: Any) -> Any | None:
        links = list(ue.agent.robot.links)
        return next(
            (
                link
                for link in links
                if "hand" in link.name.lower() and "tcp" not in link.name.lower()
            ),
            None,
        )

    def _link_point(self, link: Any, disparity: Any, left: dict[str, Any]):
        import numpy as np

        if link is None:
            return None, 0
        mask = left["segmentation"] == self._actor_id(link)
        return self._point_from_mask(disparity, mask, left["intrinsic"]), int(
            np.count_nonzero(mask)
        )

    def build_frame(self, obs: Any, env: Any, action: Any | None = None):  # noqa: ARG002
        import cv2
        import numpy as np

        ue = env.unwrapped
        scene = ue.scene
        scene.update_render(update_sensors=True, update_human_render_cameras=False)
        left = self._camera_obs(scene.sensors[self.left_uid])
        right = self._camera_obs(scene.sensors[self.right_uid])
        self.last_stereo_pair = (left["rgb"].copy(), right["rgb"].copy())

        left_gray = cv2.cvtColor(left["rgb"], cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right["rgb"], cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        disparity = self._make_matcher().compute(
            clahe.apply(left_gray), clahe.apply(right_gray)
        ).astype(np.float32) / 16.0
        disparity[disparity <= 0.0] = np.nan

        frame: dict[str, Any] = {}
        if self.emit_rgb:
            frame["rgb"] = left["rgb"].astype(np.uint8)

        left_finger, right_finger = wrapper.ManiSkillContactAdapterV1._finger_links(ue)
        left_finger_point, left_finger_pixels = self._link_point(
            left_finger, disparity, left
        )
        right_finger_point, right_finger_pixels = self._link_point(
            right_finger, disparity, left
        )
        finger_points = [
            point
            for point in (left_finger_point, right_finger_point)
            if point is not None
        ]
        gripper_source = "both_fingers" if len(finger_points) == 2 else "one_finger"
        gripper_left = (
            np.mean(np.stack(finger_points), axis=0) if finger_points else None
        )
        hand_pixels = 0
        if gripper_left is None:
            gripper_left, hand_pixels = self._link_point(
                self._hand_link(ue), disparity, left
            )
            gripper_source = "hand_fallback" if gripper_left is not None else "missing"

        center_extrinsic = self._center_extrinsic(
            left["extrinsic"], right["extrinsic"]
        )
        gripper_world = None
        gripper_center = None
        if gripper_left is not None:
            gripper_world = self._cam_to_world(gripper_left, left["extrinsic"])
            gripper_center = self._world_to_cam(gripper_world, center_extrinsic)
            self._put(frame, "gripper_cam_x", gripper_center[0])
            self._put(frame, "gripper_cam_y", gripper_center[1])
            self._put(frame, "gripper_cam_depth_m", gripper_center[2])
        if left_finger_point is not None:
            self._put(frame, "left_finger_cam_depth_m", left_finger_point[2])
        if right_finger_point is not None:
            self._put(frame, "right_finger_cam_depth_m", right_finger_point[2])

        object_actor, support_actors = self._entity_actors(ue)
        if object_actor is None:
            self.diagnostics = {
                "stereo_status": "missing_peg_hole_locator",
                "stereo_dense_valid_ratio": float(np.mean(np.isfinite(disparity))),
                "depth_fields_emitted": sum(
                    key in (wrapper.GEOMETRY_FEATURES | GRIPPER_GEOMETRY_FEATURES)
                    for key in frame
                ),
                "gripper_status": gripper_source,
                "gripper_mask_pixels": left_finger_pixels
                + right_finger_pixels
                + hand_pixels,
            }
            wrapper.validate_frame_keys(
                frame, self.allowed, context="ManiSkillStereoFeatureAdapterV2"
            )
            return frame

        object_mask = left["segmentation"] == self._actor_id(object_actor)
        object_left = self._point_from_mask(disparity, object_mask, left["intrinsic"])
        support_left_points = []
        for actor in support_actors:
            mask = left["segmentation"] == self._actor_id(actor)
            point = self._point_from_mask(disparity, mask, left["intrinsic"])
            if point is not None:
                support_left_points.append(point)
        if object_left is None or len(support_left_points) != len(support_actors):
            self.diagnostics = {
                "stereo_status": "entity_mask_or_disparity_missing",
                "stereo_dense_valid_ratio": float(np.mean(np.isfinite(disparity))),
                "object_mask_pixels": int(object_mask.sum()),
                "support_points_found": len(support_left_points),
                "depth_fields_emitted": sum(
                    key in (wrapper.GEOMETRY_FEATURES | GRIPPER_GEOMETRY_FEATURES)
                    for key in frame
                ),
                "gripper_status": gripper_source,
                "gripper_mask_pixels": left_finger_pixels
                + right_finger_pixels
                + hand_pixels,
            }
            wrapper.validate_frame_keys(
                frame, self.allowed, context="ManiSkillStereoFeatureAdapterV2"
            )
            return frame

        object_world = self._cam_to_world(object_left, left["extrinsic"])
        supports_world = [
            self._cam_to_world(point, left["extrinsic"])
            for point in support_left_points
        ]
        support_world = np.mean(np.stack(supports_world), axis=0)
        half_size = float(wrapper._to_numpy(ue.cube_half_size).reshape(-1)[0])
        goal_world = support_world + np.array([0.0, 0.0, 2.0 * half_size])
        object_center = self._world_to_cam(object_world, center_extrinsic)
        goal_center = self._world_to_cam(goal_world, center_extrinsic)
        support_center = self._world_to_cam(support_world, center_extrinsic)
        object_right = self._world_to_cam(object_world, right["extrinsic"])
        goal_left = self._world_to_cam(goal_world, left["extrinsic"])

        world_delta = object_world - goal_world
        center_delta = object_center - goal_center
        distance = float(np.linalg.norm(world_delta))
        moved = (
            0.0
            if self._initial_object_world is None
            else float(np.linalg.norm(object_world - self._initial_object_world))
        )
        speed = (
            None
            if self._previous_object_world is None
            else float(np.linalg.norm(object_world - self._previous_object_world))
        )
        distance_delta = (
            None if self._previous_distance is None else self._previous_distance - distance
        )

        self._put(frame, "object_cam_x", object_center[0])
        self._put(frame, "object_cam_y", object_center[1])
        self._put(frame, "object_cam_depth_m", object_center[2])
        self._put(frame, "goal_cam_x", goal_center[0])
        self._put(frame, "goal_cam_y", goal_center[1])
        self._put(frame, "goal_cam_depth_m", goal_center[2])
        self._put(frame, "support_cam_depth_m", support_center[2])
        self._put(frame, "object_goal_3d_dist_m", distance)
        self._put(frame, "object_goal_xy_error_m", np.linalg.norm(world_delta[:2]))
        self._put(frame, "object_goal_height_error_m", abs(world_delta[2]))
        self._put(frame, "delta_object_goal_dist_m", distance_delta)
        self._put(frame, "object_moved_from_start_m", moved)
        self._put(
            frame,
            "object_goal_cam_lateral_error_m",
            np.linalg.norm(center_delta[:2]),
        )
        self._put(frame, "object_goal_cam_depth_error_m", abs(center_delta[2]))
        self._put(frame, "left_object_depth_m", object_left[2])
        self._put(frame, "right_object_depth_m", object_right[2])
        self._put(frame, "left_goal_depth_m", goal_left[2])
        if object_center[2] > 0:
            self._put(
                frame,
                "object_pseudo_disparity_px",
                self.pseudo_focal_px * self.baseline_m / object_center[2],
            )
        if goal_center[2] > 0:
            self._put(
                frame,
                "goal_pseudo_disparity_px",
                self.pseudo_focal_px * self.baseline_m / goal_center[2],
            )
        self._put(frame, "object_speed_proxy_m_per_step", speed)
        if speed is not None:
            self._put(frame, "object_static_proxy", speed <= 0.003)

        if gripper_world is not None and gripper_center is not None:
            gripper_object_delta = gripper_center - object_center
            self._put(
                frame,
                "gripper_object_3d_dist_m",
                np.linalg.norm(gripper_world - object_world),
            )
            self._put(
                frame,
                "gripper_goal_3d_dist_m",
                np.linalg.norm(gripper_world - goal_world),
            )
            self._put(
                frame,
                "gripper_object_cam_lateral_error_m",
                np.linalg.norm(gripper_object_delta[:2]),
            )
            self._put(
                frame,
                "gripper_object_cam_depth_error_m",
                abs(gripper_object_delta[2]),
            )

        if self._initial_object_world is None:
            self._initial_object_world = object_world.copy()
        self._previous_object_world = object_world.copy()
        self._previous_distance = distance
        self.diagnostics = {
            "stereo_status": "ok",
            "stereo_dense_valid_ratio": float(np.mean(np.isfinite(disparity))),
            "object_mask_pixels": int(object_mask.sum()),
            "depth_fields_emitted": sum(
                key in (wrapper.GEOMETRY_FEATURES | GRIPPER_GEOMETRY_FEATURES)
                for key in frame
            ),
            "gripper_status": gripper_source,
            "gripper_mask_pixels": left_finger_pixels
            + right_finger_pixels
            + hand_pixels,
        }
        wrapper.validate_frame_keys(
            frame, self.allowed, context="ManiSkillStereoFeatureAdapterV2"
        )
        return frame

