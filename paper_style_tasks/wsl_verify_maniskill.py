from __future__ import annotations

import sys

import gymnasium as gym
import mani_skill
import mani_skill.envs  # noqa: F401
import mplib
import numpy
import sapien
import scipy
import torch


def main() -> None:
    print("python", sys.executable)
    print("torch", torch.__version__)
    print("numpy", numpy.__version__)
    print("scipy", scipy.__version__)
    print("gymnasium", gym.__version__)
    print("sapien", sapien.__version__)
    print("mplib", mplib.__file__)
    print("mani_skill", getattr(mani_skill, "__version__", mani_skill.__file__))

    for env_id in ["PickCube-v1", "StackCube-v1", "StackPyramid-v1", "PegInsertionSide-v1"]:
        env = gym.make(
            env_id,
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            render_mode="rgb_array",
            render_backend="cpu",
            max_episode_steps=10,
        )
        _, info = env.reset(seed=0)
        frame = env.render()
        print(env_id, "frame", getattr(frame, "shape", None), "info keys", sorted(info.keys()))
        env.close()


if __name__ == "__main__":
    main()
