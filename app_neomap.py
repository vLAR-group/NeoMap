"""Local/private interactive NeoMap workbench. SSH tunnel recommended."""
import argparse
from pathlib import Path
import sys
import uuid
import numpy as np
import gradio as gr
from neomap_release.jobs import Jobs
from neomap_release.runner import ROOT, DEFAULT_CONFIG
from neomap_release.scene import (load_scene, source_camera, orbit_camera, render,
    interpolate, trajectory, export_bundle, write_ply)


def cloud_figure(scene):
    """Use a real point-cloud plot, not a Gaussian-splat PLY viewer."""
    import plotly.graph_objects as go
    stride = max(1, int(np.ceil(len(scene['points'])/30000)))
    points = scene['points'][::stride] * [1, -1, -1]
    colors = ['rgb(%d,%d,%d)' % tuple(c) for c in scene['colors'][::stride]]
    centers = scene['poses'][:, :3, 3] * [1, -1, -1]
    figure = go.Figure([
        go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers',
                     marker=dict(size=1.5, color=colors), name='Scene', hoverinfo='skip'),
        go.Scatter3d(x=centers[:, 0], y=centers[:, 1], z=centers[:, 2], mode='markers+text',
                     marker=dict(size=5, color='orange'), text=[str(i) for i in range(len(centers))], name='Input cameras')])
    figure.update_layout(scene=dict(aspectmode='data', dragmode='orbit'), margin=dict(l=0, r=0, t=0, b=0), height=420)
    return figure


def build_app(output_root, config):
    output_root = Path(output_root).resolve()
    jobs = Jobs(output_root/'jobs')

    def available(s):
        if not s or not s.get('scene'):
            raise gr.Error('Upload images and reconstruct a scene first.')
        return load_scene(s['scene'])

    def busy(s):
        return s and s.get('job') and jobs.status(s['job'])[0] == 'running/queued'

    def rebuild(files, gpu, s):
        if busy(s):
            raise gr.Error('Finish or cancel the current job first.')
        if not files:
            raise gr.Error('Upload one or more images.')
        if len(files) > 16:
            raise gr.Error('This initial release limits reconstruction to 16 input images.')
        directory = output_root/uuid.uuid4().hex
        directory.mkdir()
        job = jobs.start(['reconstruct', str(config), str(directory), *map(str, files)], int(gpu))
        return dict(job=job, kind='scene', directory=str(directory), scene=None, keys=[], index=0), 'VGGT queued', None, None

    def poll(s):
        if not s or not s.get('job'):
            return s, 'Idle', gr.skip(), gr.skip(), gr.skip()
        status, log = jobs.status(s['job'])
        cloud, video, choices = gr.skip(), gr.skip(), gr.skip()
        if status == 'completed' and s.get('loaded_job') != s['job']:
            s = dict(s, loaded_job=s['job'])
            if s['kind'] == 'scene':
                s['scene'] = str(Path(s['directory'])/'scene.npz')
                scene = available(s)
                s['keys'] = [source_camera(scene, 0)]
                cloud = cloud_figure(scene)
                choices = gr.update(choices=[str(i) for i in range(len(scene['images']))], value='0')
            elif s['kind'] == 'generate':
                video = str(Path(s['generation'])/('prior.mp4' if s.get('cpu') else 'output.mp4'))
                s['video'] = video
            elif s['kind'] == 'review':
                cloud = cloud_figure(load_scene(Path(s['review'])/'scene.npz'))
        return s, status+'\n'+log, cloud, video, choices

    def select_source(s, index):
        scene = available(s)
        s = dict(s, index=int(index), keys=[source_camera(scene, int(index))])
        return s, scene['images'][int(index)], '1 keyframe (source); previous path reset'

    def view(s, yaw, pitch, radius, x, y, z):
        scene = available(s)
        cam = orbit_camera(scene, s['index'], yaw, pitch, radius, (x, y, z))
        h, w = scene['images'][s['index']].shape[:2]
        return render(scene, cam, w, h)[0]

    def add(s, yaw, pitch, radius, x, y, z):
        scene = available(s)
        keys = list(s['keys'])
        if len(keys) >= 81:
            raise gr.Error('At most 81 keyframes.')
        keys.append(orbit_camera(scene, s['index'], yaw, pitch, radius, (x, y, z)))
        return dict(s, keys=keys), f'{len(keys)} keyframes'

    def undo(s):
        available(s)
        keys = s['keys'][:-1] if len(s['keys']) > 1 else s['keys']
        return dict(s, keys=keys), f'{len(keys)} keyframes'

    def save(s, frames):
        scene = available(s)
        h, w = scene['images'][s['index']].shape[:2]
        traj = trajectory(w, h, source_camera(scene, s['index']), interpolate(s['keys'], int(frames)))
        directory = Path(s['directory'])/'exports'/uuid.uuid4().hex
        path = export_bundle(directory, scene, s['index'], traj, s['keys'])
        return dict(s, bundle=str(directory)), path

    def launch(s, frames, gpu, prompt, tau, cpu):
        if busy(s):
            raise gr.Error('Finish or cancel the current job first.')
        s, path = save(s, frames)
        output = Path(s['directory'])/'generations'/uuid.uuid4().hex
        command = [sys.executable, str(ROOT/'main_neomap.py'), 'bundle', '--bundle', s['bundle'],
                   '--config', str(config), '--output-dir', str(output), '--prompt', prompt, '--tau', str(tau)]
        if cpu:
            command.append('--prepare-only')
        s.update(job=jobs.start(command, int(gpu), cpu=cpu), kind='generate', generation=str(output), cpu=cpu)
        return s, 'Generation queued; progress updates below', path

    def review(s, gpu):
        if busy(s):
            raise gr.Error('Finish or cancel the current job first.')
        if not s or not s.get('video') or s.get('cpu'):
            raise gr.Error('Generate an actual Wan video first.')
        output = Path(s['directory'])/'review'/uuid.uuid4().hex
        s = dict(s, job=jobs.start(['review', str(config), str(output), s['video']], int(gpu)), kind='review', review=str(output))
        return s, 'VGGT review queued (8 evenly spaced generated frames)'

    def cancel(s):
        if s:
            jobs.cancel(s.get('job'))
        return 'Cancellation requested'

    with gr.Blocks(title='NeoMap release') as app:
        gr.Markdown('# NeoMap\nReconstruct → choose source → preview/add camera keyframes → save/generate → review. '
                    'Point-cloud dragging is for inspection; **sliders below define the generation camera**. '
                    'All work and artifacts stay on this server unless you explicitly download them.')
        state = gr.State({})
        with gr.Row():
            files = gr.File(file_count='multiple', file_types=['image'], label='Input images (1–16)')
            gpu = gr.Number(value=4, precision=0, label='Physical GPU index (must be free)')
            reconstruct_button = gr.Button('1. Reconstruct with VGGT')
        with gr.Row():
            cloud = gr.Plot(label='Scene / generated scene (independent scale)')
            preview = gr.Image(label='Selected camera RGB preview; black = invisible', height=420)
        source = gr.Dropdown(choices=['0'], value='0', label='Source image index (changing resets keyframes)')
        with gr.Row():
            yaw = gr.Slider(-90, 90, value=30, label='Horizontal orbit (degrees)')
            pitch = gr.Slider(-60, 60, value=0, label='Vertical orbit (degrees)')
            radius = gr.Slider(.2, 3, value=1, label='Radius / initial radius')
        with gr.Row():
            x = gr.Number(value=0, label='Local X translation (scene units)')
            y = gr.Number(value=0, label='Local Y translation')
            z = gr.Number(value=0, label='Local Z translation')
        controls = [state, yaw, pitch, radius, x, y, z]
        with gr.Row():
            preview_button = gr.Button('Preview camera')
            add_button = gr.Button('Confirm / add keyframe')
            undo_button = gr.Button('Remove last keyframe')
            key_status = gr.Textbox(value='Reconstruct first', label='Path')
        with gr.Row():
            frames = gr.Dropdown(choices=list(range(5, 82, 4)), value=49, label='Frames (4n+1, max 81)')
            tau = gr.Number(value=.98, label='τ')
            cpu = gr.Checkbox(value=False, label='CPU render-only test (no Wan)')
        prompt = gr.Textbox(value='A stationary scene. The camera moves smoothly.', label='Prompt')
        with gr.Row():
            save_button = gr.Button('Save camera bundle')
            generate_button = gr.Button('Generate', variant='primary')
            cancel_button = gr.Button('Cancel current job')
            load_button = gr.Button('Load generated video with VGGT')
        camera_file = gr.File(label='Exported cameras.json (full bundle stays on server)')
        status = gr.Textbox(label='Job status / recent log', lines=12)
        video = gr.Video(label='Generated video / CPU prior preview')
        # Private UI callbacks: disabling public API docs also avoids the old
        # Gradio 5.12 / newer Pydantic boolean-schema incompatibility in Gene.
        reconstruct_button.click(rebuild, [files, gpu, state], [state, status, cloud, video], api_name=False)
        source.change(select_source, [state, source], [state, preview, key_status], api_name=False)
        preview_button.click(view, controls, preview, api_name=False)
        add_button.click(add, controls, [state, key_status], api_name=False)
        undo_button.click(undo, state, [state, key_status], api_name=False)
        save_button.click(save, [state, frames], [state, camera_file], api_name=False)
        generate_button.click(launch, [state, frames, gpu, prompt, tau, cpu], [state, status, camera_file], api_name=False)
        cancel_button.click(cancel, state, status, api_name=False)
        load_button.click(review, [state, gpu], [state, status], api_name=False)
        gr.Timer(3).tick(poll, state, [state, status, cloud, video, source], api_name=False)
    return app


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(DEFAULT_CONFIG))
    p.add_argument('--output-dir', default=str(ROOT/'outputs/ui'))
    p.add_argument('--port', type=int, default=7860)
    a = p.parse_args()
    build_app(a.output_dir, Path(a.config).resolve()).queue().launch(
        server_name='127.0.0.1', server_port=a.port, share=False,
        allowed_paths=[str(Path(a.output_dir).resolve())])
