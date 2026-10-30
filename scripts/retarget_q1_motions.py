"""Retarget G1 references to Q1 using bounded foot-pose inverse kinematics.

This is a geometric bootstrap motion prior, not dynamically validated mocap.
Root translations are scaled by neutral sole height; foot displacements are
scaled around each robot's neutral stance, preserving Q1's own hip spacing.
The original frame period (20 ms) is retained. Added arm/waist-roll DoFs stay
at the Q1 neutral pose. All velocities are recomputed after retargeting.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]


class Robot:
    def __init__(self, cfg):
        self.names = cfg['asset']['dof_names']
        path = ROOT / cfg['asset']['file']
        urdf = ET.parse(path).getroot()
        # The inherited G1 URDF repeats 'meshes/' in its compiler and filenames.
        # Resolve filenames explicitly; this changes no kinematics or inertias.
        if urdf.find('mujoco') is not None:
            urdf.remove(urdf.find('mujoco'))
        ET.SubElement(ET.SubElement(urdf, 'mujoco'), 'compiler', fusestatic='false', discardvisual='false')
        for mesh in urdf.findall('.//mesh'):
            mesh.set('filename', str((path.parent / mesh.get('filename')).resolve()))
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(urdf, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.indices = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in self.names])
        self.feet = [self.model.body(n).id for n in cfg['asset']['foot_names']]
        defaults = cfg['init_state']['default_joint_angles']
        self.neutral = np.array([defaults.get(n, defaults['default']) for n in self.names])
        ids = [self.model.joint(n).id for n in self.names]
        self.limits = self.model.jnt_range[ids].copy()

    def fk(self, q):
        self.data.qpos[self.indices] = q
        mujoco.mj_forward(self.model, self.data)
        return self.data.xpos[self.feet].copy(), self.data.xmat[self.feet].reshape(2, 3, 3).copy()


def retarget(input_dir, output_dir, dt):
    cfg = yaml.safe_load((ROOT / 'envs/Q1.yaml').read_text())
    source_cfg = yaml.safe_load((ROOT / 'envs/G1.yaml').read_text())
    source, target = Robot(source_cfg), Robot(cfg)
    source_neutral, _ = source.fk(source.neutral)
    target_neutral, _ = target.fk(target.neutral)
    edges = np.array(cfg['asset']['feet_edge_pos'])
    source_edges = np.array(source_cfg['asset']['feet_edge_pos'])
    source_height = -source_neutral[:, 2].mean() - source_edges[:, 2].min()
    target_height = -target_neutral[:, 2].mean() - edges[:, 2].min()
    scale = target_height / source_height
    output_dir.mkdir(parents=True, exist_ok=True)
    header = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'v_x', 'v_y', 'v_z', 'w_x', 'w_y', 'w_z']
    header += ['q_' + n for n in target.names] + ['dq_' + n for n in target.names]
    manifest = {'source': 'data/g1', 'method': 'bounded foot-pose IK about neutral stance',
                'frame_dt': dt, 'translation_scale': float(scale),
                'neutral_base_height': float(target_height),
                'limitations': 'Geometric bootstrap only; no dynamic feasibility or kick success guarantee. New upper-body DoFs use neutral references.',
                'files': {}}
    paths = sorted(input_dir.glob('*.csv'))
    if not paths:
        raise FileNotFoundError(f'No source CSV files in {input_dir}')
    for path in paths:
        with path.open(newline='') as f:
            source_header = next(csv.reader(f))
        expected = header[:13] + ['q_' + n for n in source.names] + ['dq_' + n for n in source.names]
        if source_header != expected:
            raise ValueError(f'Unexpected G1 motion layout: {path}')
        rows = np.loadtxt(path, delimiter=',', skiprows=1)
        n = len(rows)
        result = np.zeros((n, len(header)))
        result[:, :3] = rows[:, :3] * scale
        result[:, 3:7] = Rotation.from_quat(rows[:, 3:7]).as_quat()
        positions = np.tile(target.neutral, (n, 1))
        previous = target.neutral.copy()
        errors, lifts = [], []
        for i, row in enumerate(rows):
            src_pos, src_rot = source.fk(row[13:26])
            desired_pos = target_neutral + scale * (src_pos - source_neutral)
            q = previous.copy()
            # Both robots have six leg joints per side in the same semantic order.
            for side in range(2):
                sl = slice(side * 6, side * 6 + 6)
                def residual(leg):
                    q[sl] = leg
                    p, r = target.fk(q)
                    angle = Rotation.from_matrix(src_rot[side] @ r[side].T).as_rotvec()
                    return np.r_[p[side] - desired_pos[side], .06 * angle]
                solution = least_squares(residual, np.clip(previous[sl], target.limits[sl, 0] + 1e-6, target.limits[sl, 1] - 1e-6),
                                         bounds=(target.limits[sl, 0], target.limits[sl, 1]),
                                         max_nfev=35, ftol=1e-6, xtol=1e-6, gtol=1e-6)
                q[sl] = solution.x
            q[13] = np.clip(row[25], *target.limits[13])  # waist yaw
            q[14:] = target.neutral[14:]
            p, r = target.fk(q)
            errors.extend(np.linalg.norm(p - desired_pos, axis=1).tolist())
            # Keep the lowest sole corner on/above the floor under root tilt.
            base_rotation = Rotation.from_quat(result[i, 3:7])
            corners = p[:, None, :] + np.einsum('bij,kj->bki', r, edges)
            minimum_z = base_rotation.apply(corners.reshape(-1, 3))[:, 2].min()
            lift = max(0., -minimum_z - result[i, 2])
            result[i, 2] += lift
            lifts.append(lift)
            positions[i] = q
            previous = q.copy()
        result[:, 13:35] = positions
        result[:, 35:57] = np.gradient(positions, dt, axis=0)
        rotations = Rotation.from_quat(result[:, 3:7])
        result[:, 7:10] = rotations.inv().apply(np.gradient(result[:, :3], dt, axis=0))
        angular = (rotations[:-1].inv() * rotations[1:]).as_rotvec() / dt
        result[:-1, 10:13] = angular
        result[-1, 10:13] = angular[-1]
        if not np.isfinite(result).all():
            raise ValueError(f'Non-finite retarget result: {path}')
        # Exclude entire clips with large unreachable foot targets. This keeps
        # adjacent AMP samples continuous and avoids stitching across bad frames.
        accepted = max(errors) <= .035
        out = output_dir / path.name
        if accepted:
            np.savetxt(out, result, delimiter=',', header=','.join(header), comments='', fmt='%.7f')
        elif out.exists():
            out.unlink()
        manifest['files'][path.name] = {
            'accepted': bool(accepted),
            'frames': n, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'output_sha256': hashlib.sha256(out.read_bytes()).hexdigest() if accepted else None,
            'foot_position_rmse_m': float(np.sqrt(np.mean(np.square(errors)))),
            'foot_position_max_error_m': float(max(errors)),
            'max_floor_lift_m': float(max(lifts)),
        }
        print(f'{path.name}: {n} frames, foot RMSE {manifest["files"][path.name]["foot_position_rmse_m"]:.4f} m, accepted={accepted}', flush=True)
    (output_dir / 'retarget_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/g1')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/q1')
    parser.add_argument('--dt', type=float, default=.02)
    args = parser.parse_args()
    if args.dt <= 0:
        parser.error('--dt must be positive')
    retarget(args.input, args.output, args.dt)
