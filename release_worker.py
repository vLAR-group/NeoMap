"""Private worker: GPU lock + reconstruction/video review or CLI dispatch."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', required=True, type=int)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    lock = None
    if not a.cpu:
        lock = open(f'/tmp/neomap-release-{os.getuid()}-gpu-{a.gpu}.lock', 'a')
        print('STAGE: waiting for release GPU lock', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        stats = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        used, utilization = map(int, stats.strip().split(','))
        if used > 1024 or utilization > 5:
            raise RuntimeError(f'GPU {a.gpu} is busy ({used} MiB, {utilization}%); select a free GPU and retry.')
    if command[0] in ('reconstruct', 'review'):
        from neomap_release.runner import reconstruct
        from neomap_release.scene import save_scene, write_ply
        mode, config, output, *paths = command
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        if mode == 'review':
            import imageio.v2 as imageio
            import numpy as np
            from PIL import Image
            reader = imageio.get_reader(paths[0])
            try:
                indices = np.unique(np.linspace(0, reader.count_frames()-1, 8).astype(int))
                paths = []
                for i in indices:
                    path = output/f'frame_{i:04d}.png'
                    Image.fromarray(reader.get_data(int(i))).save(path)
                    paths.append(str(path))
            finally:
                reader.close()
        scene = reconstruct(paths, config)
        save_scene(output/'scene.npz', scene)
        write_ply(output/'scene.ply', scene)
        print('DONE: reconstruction saved', flush=True)
    else:
        result = subprocess.run(command, check=False)
        sys.exit(result.returncode)


if __name__ == '__main__':
    main()
