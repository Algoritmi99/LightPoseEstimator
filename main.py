import logging
from pathlib import Path

import trimesh

from LightPoseEstim.dataloader import DataLoader
from LightPoseEstim.keypoint_selector import MeshKeypointSelector


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    loader = DataLoader(Path("/home/algoritmi/Projects/LightPoseEstimator/data/Astrobee"))
    print(len(loader.get_dataset()))

    mesh = trimesh.load_mesh('~/Downloads/Astrobee v5_real.stl')
    kp_selector = MeshKeypointSelector()
    kp = kp_selector.select_keypoints(mesh)
    kp_selector.show_keypoints(mesh, kp)



if __name__ == '__main__':
    main()
