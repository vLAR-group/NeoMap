"""CPU geometry. All poses are OpenCV camera-to-world; depths are camera Z."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import PchipInterpolator


def frame_count(n):
    if isinstance(n, bool) or int(n) != n or not 5 <= n <= 81 or (n - 1) % 4:
        raise ValueError('Frame count must be 4n+1, between 5 and 81.')
    return int(n)


def camera(k, pose):
    k, pose = np.asarray(k, dtype=float), np.asarray(pose, dtype=float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Invalid camera intrinsics.')
    if not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1]) > 1e-6 or abs(k[1, 0]) > 1e-6:
        raise ValueError('Expected zero-skew pinhole intrinsics.')
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError('Invalid camera_to_world matrix.')
    r = pose[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(r), 1, atol=1e-4):
        raise ValueError('Camera rotation must be right-handed and orthonormal.')
    return {'intrinsics': k.tolist(), 'camera_to_world': pose.tolist()}


def trajectory(width, height, source, targets, fps=8):
    frame_count(len(targets))
    if int(width) != width or int(height) != height or min(width, height) <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid image size or fps.')
    return dict(version=1, coordinate_convention='opencv', pose_convention='camera_to_world',
                depth_convention='camera_z', width=int(width), height=int(height), fps=float(fps),
                source_camera=camera(source['intrinsics'], source['camera_to_world']),
                target_cameras=[camera(c['intrinsics'], c['camera_to_world']) for c in targets])


def read_trajectory(path):
    data = json.loads(Path(path).read_text())
    expected = dict(version=1, coordinate_convention='opencv', pose_convention='camera_to_world', depth_convention='camera_z')
    if any(data.get(k) != v for k, v in expected.items()):
        raise ValueError('Unsupported camera format/conventions; see docs/RELEASE.md.')
    return trajectory(data['width'], data['height'], data['source_camera'], data['target_cameras'], data.get('fps', 8))


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False))


def unproject(rgb, depth, cam):
    h, w = depth.shape
    if rgb.shape != (h, w, 3):
        raise ValueError('RGB and depth sizes must match; no implicit depth resizing.')
    y, x = np.mgrid[:h, :w]
    k = np.asarray(cam['intrinsics'])
    xyz = np.stack(((x-k[0, 2])/k[0, 0]*depth, (y-k[1, 2])/k[1, 1]*depth, depth), -1)
    good = np.isfinite(xyz).all(-1) & (depth > 0)
    pose = np.asarray(cam['camera_to_world'])
    return xyz[good] @ pose[:3, :3].T + pose[:3, 3], rgb[good]


def explicit_scene(image, depth_path, traj, depth_scale=1):
    rgb = np.asarray(Image.open(image).convert('RGB'))
    if depth_scale <= 0 or not np.isfinite(depth_scale):
        raise ValueError('depth_scale must be positive and finite.')
    depth = np.load(depth_path, allow_pickle=False) if str(depth_path).lower().endswith('.npy') else np.asarray(Image.open(depth_path))
    depth = np.asarray(depth, dtype=np.float32) * depth_scale
    if depth.shape != (traj['height'], traj['width']) or rgb.shape[:2] != depth.shape:
        raise ValueError('Image, depth and cameras.json dimensions must match.')
    pts, colors = unproject(rgb, depth, traj['source_camera'])
    if len(pts) == 0:
        raise ValueError('Depth has no valid positive values.')
    return dict(images=rgb[None], depths=depth[None], intrinsics=np.array([traj['source_camera']['intrinsics']]),
                poses=np.array([traj['source_camera']['camera_to_world']]), points=pts, colors=colors)


def save_scene(path, scene):
    np.savez_compressed(path, **scene)


def load_scene(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def source_camera(scene, index=0):
    return camera(scene['intrinsics'][index], scene['poses'][index])


def pivot(scene, index=0):
    depth = scene['depths'][index]
    h, w = depth.shape
    central = depth[h//4:3*h//4, w//4:3*w//4]
    valid = central[np.isfinite(central) & (central > 0)]
    if not valid.size:
        valid = depth[np.isfinite(depth) & (depth > 0)]
    if not valid.size:
        raise ValueError('No valid depth for orbit pivot.')
    # Optical-axis point at the central-region median Z depth.
    p = scene['poses'][index]
    return p[:3, 3] + p[:3, 2] * np.median(valid)


def orbit_camera(scene, index=0, yaw=0, pitch=0, radius_scale=1, offset=(0, 0, 0)):
    if radius_scale <= 0:
        raise ValueError('Radius scale must be positive.')
    start = scene['poses'][index]
    center = pivot(scene, index)
    local = Rotation.from_euler('yx', [yaw, pitch], degrees=True).as_matrix()
    rotation = start[:3, :3] @ local @ start[:3, :3].T
    pose = np.eye(4)
    pose[:3, :3] = rotation @ start[:3, :3]
    pose[:3, 3] = center + rotation @ ((start[:3, 3]-center)*radius_scale) + start[:3, :3] @ np.asarray(offset)
    return camera(scene['intrinsics'][index], pose)


def orbit_trajectory(scene, index=0, angle=30, frames=49, radius_scale=1, fps=8):
    frame_count(frames)
    targets = []
    for t in np.linspace(0, 1, frames):
        t = t*t*(3-2*t)
        targets.append(orbit_camera(scene, index, angle*t, radius_scale=1+(radius_scale-1)*t))
    h, w = scene['images'][index].shape[:2]
    return trajectory(w, h, source_camera(scene, index), targets, fps)


def interpolate(keys, frames):
    frame_count(frames)
    if len(keys) < 2:
        raise ValueError('Add at least one target keyframe after the source camera.')
    if len(keys) > frames:
        raise ValueError('More keyframes than output frames.')
    poses = np.array([camera(c['intrinsics'], c['camera_to_world'])['camera_to_world'] for c in keys])
    positions = poses[:, :3, 3]
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    angles = (rotations[:-1].inv()*rotations[1:]).magnitude()
    weights = np.maximum(lengths + angles * max(lengths.max(), 1e-3), 1e-5)
    # Allocate integer frame intervals so every keyframe is reached exactly.
    intervals = np.ones(len(keys)-1, dtype=int)
    for _ in range(frames-1-len(intervals)):
        intervals[np.argmax(weights/intervals)] += 1
    knots = np.r_[0, np.cumsum(intervals)].astype(float)
    position_curve = PchipInterpolator(knots, positions, axis=0)
    slerp = Slerp(knots, rotations)
    result = []
    for i, count in enumerate(intervals):
        for step in range(count):
            u = step/count
            u = u*u*(3-2*u)  # zero velocity at keys, including rotation
            t = knots[i]+u*count
            pose = np.eye(4)
            pose[:3, 3], pose[:3, :3] = position_curve(t), slerp(t).as_matrix()
            k = (1-u)*np.asarray(keys[i]['intrinsics']) + u*np.asarray(keys[i+1]['intrinsics'])
            result.append(camera(k, pose))
    result.append(keys[-1])
    return result


def render(scene, cam, width, height, splat_radius=1):
    """Conservative z-buffer point splats, without inpainting invisible pixels."""
    pose, k = np.asarray(cam['camera_to_world']), np.asarray(cam['intrinsics'])
    pts = (scene['points']-pose[:3, 3]) @ pose[:3, :3]
    valid = np.isfinite(pts).all(1) & (pts[:, 2] > 1e-5)
    pts, colors = pts[valid], scene['colors'][valid]
    screen = pts @ k.T
    uv = np.rint(screen[:, :2]/screen[:, 2:]).astype(np.int64)
    zbuf = np.full(width*height, np.inf)
    rgb = np.zeros((width*height, 3), np.uint8)
    # Far-to-near sorting plus unique nearest sample per pixel.
    order = np.argsort(pts[:, 2], kind='stable')
    for dy in range(-splat_radius, splat_radius+1):
        for dx in range(-splat_radius, splat_radius+1):
            x, y = uv[order, 0]+dx, uv[order, 1]+dy
            good = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            indices, candidates = (y[good]*width+x[good]), order[good]
            unique, first = np.unique(indices, return_index=True)
            candidates = candidates[first]
            closer = pts[candidates, 2] < zbuf[unique]
            dst, src = unique[closer], candidates[closer]
            zbuf[dst], rgb[dst] = pts[src, 2], colors[src]
    return rgb.reshape(height, width, 3), np.isfinite(zbuf).reshape(height, width)


def write_ply(path, scene, max_points=100000):
    # Display only: flip Y/Z to a conventional viewer up-axis. Scene stays OpenCV.
    stride = max(1, int(np.ceil(len(scene['points'])/max_points)))
    xyz = scene['points'][::stride] * [1, -1, -1]
    colors = scene['colors'][::stride]
    with open(path, 'w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        np.savetxt(f, np.c_[xyz, colors], fmt=['%.6f']*3+['%d']*3)


def export_bundle(directory, scene, index, traj, keys=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(scene['images'][index]).save(directory/'reference.png')
    np.save(directory/'depth.npy', scene['depths'][index])
    save_scene(directory/'scene.npz', scene)
    write_json(directory/'cameras.json', traj)
    write_json(directory/'keyframes.json', keys or [])
    write_json(directory/'scene_metadata.json', dict(source_index=index, coordinate_convention='opencv', scale='shared arbitrary scene units'))
    return str(directory/'cameras.json')
