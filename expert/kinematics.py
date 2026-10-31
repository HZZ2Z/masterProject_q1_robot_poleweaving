"""Batched URDF forward kinematics for observations immediately after reset."""
import math
import xml.etree.ElementTree as ET

import torch
from .torch_math import quat_mul, quat_rotate


class ResetKinematics:
    def __init__(self, urdf_path, dof_names, frame_names, device):
        urdf = ET.parse(urdf_path).getroot()
        joints = {j.find('child').get('link'): j for j in urdf.findall('joint')}
        links = {link.get('name') for link in urdf.findall('link')}
        self.frames = frame_names
        required = set()
        roots = set()
        for frame in frame_names:
            link = frame
            while link in joints:
                required.add(link)
                link = joints[link].find('parent').get('link')
            if link not in links:
                raise ValueError(f'Unknown URDF frame: {frame}')
            roots.add(link)
        if len(roots) != 1:
            raise ValueError('Reset frames must share one URDF root')
        self.root = roots.pop()
        indices = {name: i for i, name in enumerate(dof_names)}
        self.edges = []
        pending = set(required)
        visited = {self.root}
        while pending:
            ready = sorted(link for link in pending if joints[link].find('parent').get('link') in visited)
            if not ready:
                raise ValueError('Reset frames require a connected URDF tree')
            for child in ready:
                joint = joints[child]
                origin = joint.find('origin')
                xyz = [float(v) for v in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split()]
                r, p, y = [float(v) / 2 for v in (origin.get('rpy', '0 0 0') if origin is not None else '0 0 0').split()]
                sr, cr, sp, cp, sy, cy = math.sin(r), math.cos(r), math.sin(p), math.cos(p), math.sin(y), math.cos(y)
                rotation = [sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                            cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy]
                kind = joint.get('type')
                if kind not in ('fixed', 'revolute', 'continuous'):
                    raise ValueError(f'Unsupported reset joint type: {kind}')
                axis_node = joint.find('axis')
                axis = torch.tensor([float(v) for v in (axis_node.get('xyz') if axis_node is not None else '1 0 0').split()], device=device)
                index = None if kind == 'fixed' else indices[joint.get('name')]
                self.edges.append((joint.find('parent').get('link'), child,
                                   torch.tensor(xyz, device=device), torch.tensor(rotation, device=device),
                                   axis / axis.norm().clamp_min(1e-12), index))
                pending.remove(child)
                visited.add(child)

    def __call__(self, dof_pos):
        n = dof_pos.shape[0]
        zero = dof_pos.new_zeros(n, 3)
        identity = dof_pos.new_zeros(n, 4)
        identity[:, 3] = 1
        states = {self.root: (zero, identity)}
        for parent, child, xyz, rotation, axis, index in self.edges:
            pos, quat = states[parent]
            child_pos = pos + quat_rotate(quat, xyz.expand(n, -1))
            child_quat = quat_mul(quat, rotation.expand(n, -1))
            if index is not None:
                half = dof_pos[:, index:index+1] * .5
                joint_quat = torch.cat((half.sin() * axis, half.cos()), dim=-1)
                child_quat = quat_mul(child_quat, joint_quat)
            states[child] = (child_pos, child_quat)
        return {name: states[name] for name in self.frames}
