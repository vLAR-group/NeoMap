"""Single-server subprocess jobs; a shared advisory lock per physical GPU."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import uuid
from .runner import ROOT


class Jobs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.lock = threading.Lock()

    def start(self, command, gpu, cpu=False):
        if not str(gpu).isdigit():
            raise ValueError('GPU must be a physical nonnegative device index.')
        with self.lock:
            job_id = uuid.uuid4().hex
            directory = self.root/job_id
            directory.mkdir()
            log = open(directory/'job.log', 'w')
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
            # This worker waits in a queue when another release job holds the lock.
            args = [sys.executable, str(ROOT/'release_worker.py'), '--gpu', str(gpu)]
            if cpu:
                args.append('--cpu')
            args += ['--', *command]
            try:
                proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            finally:
                log.close()
            self.jobs[job_id] = (proc, directory)
            return job_id

    def status(self, job_id):
        if not job_id:
            return 'idle', ''
        proc, directory = self.jobs[job_id]
        code = proc.poll()
        state = 'running/queued' if code is None else ('completed' if code == 0 else f'failed ({code})')
        with open(directory/'job.log', 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell()-6000))
            tail = f.read().decode(errors='replace')
        return state, tail

    def cancel(self, job_id):
        if job_id and self.jobs[job_id][0].poll() is None:
            os.killpg(self.jobs[job_id][0].pid, signal.SIGTERM)

