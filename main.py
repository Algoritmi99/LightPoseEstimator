import trimesh

from LightPoseEstim.keypoint_selector import MeshKeypointSelector


def main():
    print(f'Hello World')
    mesh = trimesh.load_mesh('~/Downloads/Astrobee v5_real.stl')
    kp_selector = MeshKeypointSelector()
    kp = kp_selector.select_keypoints(mesh)
    kp_selector.show_keypoints(mesh, kp)


if __name__ == '__main__':
    main()
