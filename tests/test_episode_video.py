from pathlib import Path
from types import SimpleNamespace
import numpy as np

from utils.episode_video import EpisodeVideoRecorder


def test_video_frame_clock_terminal_capture_and_interrupted_episode(monkeypatch,tmp_path):
    import imageio.v2 as imageio
    import mujoco
    writers=[]

    class Writer:
        def __init__(self,path):
            Path(path).touch();self.frames=[];self.closed=False
        def append_data(self,frame):self.frames.append(frame.copy())
        def close(self):self.closed=True

    def writer(path,**kwargs):
        assert kwargs['fps']==50 and kwargs['codec']=='libx264'
        item=Writer(path);writers.append(item);return item

    class Renderer:
        def __init__(self,*args,**kwargs):self.scene=None;self.closed=False
        def update_scene(self,data,camera):self.marker=data.marker
        def render(self):return np.full((16,16,3),self.marker,dtype=np.uint8)
        def close(self):self.closed=True

    monkeypatch.setattr(imageio,'get_writer',writer)
    monkeypatch.setattr(mujoco,'Renderer',Renderer)
    data=SimpleNamespace(time=0.,marker=10,sensor=lambda name:SimpleNamespace(data=np.zeros(3)))
    sim=SimpleNamespace(model=SimpleNamespace(vis=SimpleNamespace(global_=SimpleNamespace(offwidth=64,offheight=64))),data=data)
    rec=EpisodeVideoRecorder(sim,tmp_path)
    rec.capture()
    data.time=.001;data.marker=11;rec.capture()  # too early for another frame
    data.time=.02;data.marker=12;rec.capture()
    data.time=.03;data.marker=13;rec.finish_episode('fall')
    data.time=.002;data.marker=20;rec.capture()  # next episode, reset simulation clock
    data.time=.022;data.marker=21;rec.capture()
    rec.close('interrupted')
    assert [[int(f[0,0,0]) for f in w.frames] for w in writers]==[[10,12,13],[20,21]]
    assert all(w.closed for w in writers) and rec.renderer.closed
    assert len(list(tmp_path.glob('*episode_000001_fall.mp4')))==1
    assert len(list(tmp_path.glob('*episode_000002_interrupted.mp4')))==1
    assert not list(tmp_path.glob('*.partial.mp4'))
