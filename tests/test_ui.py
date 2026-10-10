import tempfile
import unittest
import time
from pathlib import Path
from app_neomap import build_app, cloud_figure
from neomap_release.runner import DEFAULT_CONFIG
from neomap_release.scene import save_scene, source_camera, read_trajectory
from tests.test_release import synthetic


class UiTests(unittest.TestCase):
    def test_cloud_plot(self):
        figure = cloud_figure(synthetic())
        self.assertEqual(len(figure.data), 2)
        self.assertIn('scatter3d', figure.to_json())

    def test_edit_preview_export_callbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            app = build_app(directory, DEFAULT_CONFIG)
            # Test the actual callbacks bound to the Gradio components, offline.
            callbacks = {f.fn.__name__: f.fn for f in app.fns.values() if f.fn}
            self.assertEqual(app.get_api_info()['named_endpoints'], {})
            scene = synthetic()
            scene_path = Path(directory)/'scene.npz'
            save_scene(scene_path, scene)
            state = dict(scene=str(scene_path), directory=directory, index=0, keys=[source_camera(scene)])
            preview = callbacks['view'](state, 30, 0, 1, 0, 0, 0)
            self.assertEqual(preview.shape, (48, 64, 3))
            state, _ = callbacks['add'](state, 30, 0, 1, 0, 0, 0)
            state, path = callbacks['save'](state, 81)
            self.assertEqual(len(read_trajectory(path)['target_cameras']), 81)
            self.assertTrue((Path(state['bundle'])/'scene.npz').exists())
            state, _ = callbacks['undo'](state)
            self.assertEqual(len(state['keys']), 1)

    def test_background_cpu_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            app = build_app(directory, DEFAULT_CONFIG)
            callbacks = {f.fn.__name__: f.fn for f in app.fns.values() if f.fn}
            scene = synthetic()
            scene_path = Path(directory)/'scene.npz'
            save_scene(scene_path, scene)
            state = dict(scene=str(scene_path), directory=directory, index=0, keys=[source_camera(scene)])
            state, _ = callbacks['add'](state, 30, 0, 1, 0, 0, 0)
            state, _, _ = callbacks['launch'](state, 5, 4, 'Test', .98, True)
            try:
                for _ in range(100):
                    result = callbacks['poll'](state)
                    state, status = result[:2]
                    if not status.startswith('running/queued'):
                        break
                    time.sleep(.1)
                self.assertTrue(status.startswith('completed'), status)
                self.assertTrue(Path(state['video']).is_file())
            finally:
                callbacks['cancel'](state)


if __name__ == '__main__':
    unittest.main()
