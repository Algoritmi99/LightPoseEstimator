import logging
import trimesh

from LightPoseEstim.keypoint_selector import MeshKeypointSelector


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print(f'Hello World')
    print(trimesh.available_formats())
    mesh = trimesh.load_mesh('~/Downloads/Astrobee v5_real.stl')
    kp_selector = MeshKeypointSelector()
    kp = kp_selector.select_keypoints(mesh)
    kp_selector.show_keypoints(mesh, kp)



if __name__ == '__main__':
    main()
