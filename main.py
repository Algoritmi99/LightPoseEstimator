import logging
from pathlib import Path

import trimesh

from LightPoseEstim.dataloader import DataLoader
from LightPoseEstim.keypoint_selector import MeshKeypointSelector


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    loader = DataLoader(Path("/home/algoritmi/Projects/LightPoseEstimator/data/Astrobee"))
    print(len(loader.get_dataset()))

    kp_selector = MeshKeypointSelector()
    kp = kp_selector.select_keypoints(loader.mesh)
    kp_selector.show_keypoints(loader.mesh, kp)



if __name__ == '__main__':
    main()
