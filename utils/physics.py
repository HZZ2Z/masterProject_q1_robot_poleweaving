"""Torch views of MuJoCo Warp state, with native MuJoCo as CPU reference."""
import torch


class Physics:
    def __init__(self, model, count, device, threads=8):
        self.gpu = torch.device(device).type == 'cuda'
        self.graphs = {}
        self.views = {}
        if not self.gpu:
            from mujoco.batch import Batch
            self.cpu = Batch(model, count, nthread=threads)
            return
        import warp as wp
        import mujoco_warp as mjw
        self.wp, self.mjw = wp, mjw
        wp.init()
        self.torch_stream = torch.cuda.Stream(device=device)
        self.torch_stream.wait_stream(torch.cuda.current_stream(device))
        torch.cuda.set_stream(self.torch_stream)
        self.stream = wp.stream_from_torch(self.torch_stream)
        fields = ('body_mass', 'body_inertia', 'body_ipos', 'body_subtreemass',
                  'body_invweight0', 'dof_invweight0', 'actuator_acc0', 'geom_friction')
        with wp.ScopedStream(self.stream):
            self.model = mjw.put_model(model, batch_sizes={k: count for k in fields})
            self.data = mjw.make_data(model, nworld=count, nconmax=96, njmax=384)
        self.device = device

    def bind(self, name):
        if not self.gpu:
            return self._cpu_view(name, False)
        return self.wp.to_torch(getattr(self.data, name))

    def expand(self, name):
        if not self.gpu:
            return self._cpu_view(name, True)
        return self.wp.to_torch(getattr(self.model, name))

    def _cpu_view(self, name, model):
        if name not in self.views:
            native = self.cpu.expand(name) if model else self.cpu.bind(name)
            tensor = torch.from_numpy(native.copy() if not native.flags.writeable else native)
            self.views[name] = (native, tensor.float() if tensor.is_floating_point() else tensor)
        return self.views[name][1]

    def _cpu_call(self, name, ids=None):
        for native, tensor in self.views.values():
            if native.flags.writeable:
                native[:] = tensor.numpy()
        getattr(self.cpu, name)(None if ids is None else ids.cpu().numpy())
        for native, tensor in self.views.values():
            tensor.copy_(torch.from_numpy(native.copy() if not native.flags.writeable else native))

    def _run(self, name):
        wp = self.wp
        with wp.ScopedStream(self.stream):
            if name not in self.graphs:
                # Compile outside capture, then record without executing twice.
                getattr(self.mjw, name)(self.model, self.data)
                with wp.ScopedCapture(stream=self.stream) as capture:
                    getattr(self.mjw, name)(self.model, self.data)
                self.graphs[name] = capture.graph
            else:
                wp.capture_launch(self.graphs[name])

    def forward(self, ids=None):
        if not self.gpu:
            self._cpu_call('forward', ids)
        else:
            self._run('forward')

    def step(self):
        if not self.gpu:
            self._cpu_call('step')
        else:
            self._run('step')

    def set_const(self, ids=None):
        if not self.gpu:
            self._cpu_call('set_const', ids)
        else:
            self._run('set_const')

    def check(self):
        if self.gpu and self.bind('overflow').any().item():
            raise RuntimeError('MuJoCo Warp contact/constraint capacity exceeded')
