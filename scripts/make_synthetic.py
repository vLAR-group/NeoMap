"""Generate tiny deterministic inputs on the server, without model downloads."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_release import synthetic
from neomap_release.scene import orbit_trajectory, export_bundle

p = argparse.ArgumentParser()
p.add_argument('--output-dir', default='outputs/synthetic')
a = p.parse_args()
s = synthetic()
export_bundle(a.output_dir, s, 0, orbit_trajectory(s, angle=15, frames=5))
print(Path(a.output_dir).resolve())
