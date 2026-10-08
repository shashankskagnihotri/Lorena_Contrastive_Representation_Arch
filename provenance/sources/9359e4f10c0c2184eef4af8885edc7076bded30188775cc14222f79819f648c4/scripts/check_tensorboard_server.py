"""Bounded localhost-only TensorBoard startup check on an allocated node."""
import os, socket, subprocess, time, urllib.request
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sparse_contrast.common import ROOT,atomic_json
with socket.socket() as probe:
    probe.bind(('127.0.0.1',6006))
path=ROOT/'outputs/checks/tensorboard-server.log';path.parent.mkdir(exist_ok=True,parents=True)
with path.open('w') as log:
    process=subprocess.Popen([str(Path(sys.executable).parent/'tensorboard'),'--logdir',str(ROOT/'outputs/tensorboard'),'--host','127.0.0.1','--port','6006'],stdout=log,stderr=subprocess.STDOUT)
    try:
        for attempt in range(60):
            if process.poll() is not None:raise RuntimeError('TensorBoard exited; inspect '+str(path))
            try:
                with urllib.request.urlopen('http://127.0.0.1:6006/',timeout=1) as response:
                    if response.status==200:break
            except (urllib.error.URLError,TimeoutError):time.sleep(1)
        else:raise RuntimeError('TensorBoard startup timed out')
        atomic_json(ROOT/'outputs/checks/tensorboard-server.json',{'passed':True,'host':socket.gethostname(),'bind':'127.0.0.1','port':6006,'status':200,'server_stopped_after_check':True})
    finally:
        process.terminate()
        try:process.wait(timeout=20)
        except subprocess.TimeoutExpired:process.kill();process.wait()
