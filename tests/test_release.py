import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from neomap_release.scene import *


def synthetic():
    h, w = 48, 64
    y, x = np.mgrid[:h, :w]
    rgb = np.stack((x*4, y*5, (x+y)%2*150+60), -1).astype(np.uint8)
    depth = np.full((h, w), 3, np.float32)
    cam = camera([[60, 0, w/2], [0, 60, h/2], [0, 0, 1]], np.eye(4))
    points, colors = unproject(rgb, depth, cam)
    return dict(images=rgb[None], depths=depth[None], intrinsics=np.array([cam['intrinsics']]),
                poses=np.eye(4)[None], points=points, colors=colors)


class GeometryTests(unittest.TestCase):
    def test_frame_rules(self):
        for n in [5, 49, 81]:
            self.assertEqual(frame_count(n), n)
        for n in [0, 1, 48, 82, 85, 5.2]:
            with self.assertRaises(ValueError):
                frame_count(n)

    def test_identity_render(self):
        s = synthetic()
        rgb, mask = render(s, source_camera(s), 64, 48, 0)
        np.testing.assert_array_equal(rgb, s['images'][0])
        self.assertTrue(mask.all())

    def test_camera_translation_direction(self):
        s = synthetic()
        cam = source_camera(s)
        cam['camera_to_world'][0][3] = .3
        rgb, mask = render(s, cam, 64, 48, 0)
        np.testing.assert_array_equal(rgb[:, :58], s['images'][0, :, 6:])
        self.assertFalse(mask[:, 58:].any())

    def test_nearest_depth_wins(self):
        s = dict(points=np.array([[0, 0, 3], [0, 0, 2]]), colors=np.array([[255, 0, 0], [0, 255, 0]], np.uint8))
        rgb, mask = render(s, camera(np.eye(3), np.eye(4)), 8, 8, 0)
        np.testing.assert_array_equal(rgb[0, 0], [0, 255, 0])

    def test_orbit_and_interpolation(self):
        s = synthetic()
        for n in [5, 49, 81]:
            traj = orbit_trajectory(s, frames=n, radius_scale=1.4)
            np.testing.assert_allclose(traj['target_cameras'][0]['camera_to_world'], np.eye(4), atol=1e-10)
            keys = [source_camera(s), orbit_camera(s, yaw=30), orbit_camera(s, yaw=60)]
            path = interpolate(keys, n)
            self.assertEqual(len(path), n)
            for key in keys:
                self.assertTrue(any(np.allclose(c['camera_to_world'], key['camera_to_world']) for c in path))
            for c in path:
                camera(c['intrinsics'], c['camera_to_world'])

    def test_roundtrip(self):
        s = synthetic()
        traj = orbit_trajectory(s, frames=5)
        with tempfile.TemporaryDirectory() as directory:
            export_bundle(directory, s, 0, traj)
            p = Path(directory)
            loaded = read_trajectory(p/'cameras.json')
            self.assertEqual(loaded, traj)
            recovered = explicit_scene(p/'reference.png', p/'depth.npy', loaded)
            np.testing.assert_allclose(recovered['points'], s['points'])
            np.testing.assert_allclose(load_scene(p/'scene.npz')['points'], s['points'])

    def test_reject_bad_camera(self):
        p = np.eye(4)
        p[0, 0] = -1
        with self.assertRaises(ValueError):
            camera(np.eye(3), p)

    def test_invalid_depth(self):
        s = synthetic()
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            Image.fromarray(s['images'][0]).save(p/'image.png')
            np.save(p/'depth.npy', np.zeros((48, 64)))
            with self.assertRaises(ValueError):
                explicit_scene(p/'image.png', p/'depth.npy', orbit_trajectory(s, frames=5))


if __name__ == '__main__':
    unittest.main()
