"""Focused CPU checks for the G1 13-DoF training assets.

Run with the Isaac Gym environment: python -m unittest discover -s tests -q
"""
import csv
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from isaacgym import gymapi  # Must precede torch.
from isaacgym.torch_utils import quat_mul, quat_rotate, quat_rotate_inverse
import mujoco
import numpy as np
import pinocchio as pin
import torch
import yaml

from envs.g1 import G1
from play_mujoco_g1 import quat_mul as mujoco_quat_mul
from utils.kinematics import ResetKinematics
from utils.motion import Motion


class G1TrainingAssetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = yaml.safe_load(Path('envs/G1.yaml').read_text())

    def test_robot_models_keep_only_legs_and_waist_yaw(self):
        cfg = self.cfg
        names = cfg['asset']['dof_names']
        self.assertEqual(len(names), 13)
        root = ET.parse(cfg['asset']['file']).getroot()
        joints = {j.get('name'): j for j in root.findall('joint')}
        self.assertEqual([n for n, j in joints.items() if j.get('type') != 'fixed'], names)
        surviving_bodies = {cfg['asset']['base_name']} | {
            j.find('child').get('link') for j in joints.values() if j.get('type') != 'fixed'
        }
        self.assertIn(cfg['asset']['head_name'], surviving_bodies)
        self.assertTrue(set(cfg['asset']['foot_names']) <= surviving_bodies)
        for token in cfg['rewards']['penalize_contacts_on']:
            self.assertTrue(any(token in body for body in surviving_bodies), token)
        fixed = ['waist_roll_joint', 'waist_pitch_joint'] + [
            f'{side}_{joint}_joint'
            for side in ('left', 'right')
            for joint in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw',
                          'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')
        ]
        self.assertTrue(all(joints[name].get('type') == 'fixed' for name in fixed))
        for name, axis, angle in [('left_shoulder_roll_joint', 0, -.07925),
                                  ('right_shoulder_roll_joint', 0, .07925),
                                  ('left_elbow_joint', 1, .3),
                                  ('right_elbow_joint', 1, .3)]:
            rpy = [float(x) for x in joints[name].find('origin').get('rpy').split()]
            self.assertAlmostEqual(rpy[axis], angle)
        for foot in cfg['asset']['foot_names']:
            collision = root.find(f"link[@name='{foot}']/collision")
            self.assertEqual(len(root.find(f"link[@name='{foot}']").findall('collision')), 1)
            self.assertEqual(collision.find('geometry/box').get('size'), '0.17 0.06 0.01')
        model = mujoco.MjModel.from_xml_path(cfg['asset']['mujoco_file'])
        self.assertEqual([model.actuator(i).name for i in range(model.nu)], names)

    def test_mujoco_reset_yaw_composition(self):
        quarter_turn = np.array([0., 0., np.sqrt(.5), np.sqrt(.5)])
        np.testing.assert_allclose(mujoco_quat_mul(quarter_turn, quarter_turn),
                                   [0., 0., 1., 0.], atol=1e-15)

    def test_mujoco_soccer_scene_matches_g1_training_frame(self):
        cfg = self.cfg
        model = mujoco.MjModel.from_xml_path(cfg['asset']['mujoco_file'])
        self.assertEqual(model.nu, 13)
        np.testing.assert_allclose(model.dof_armature[6:19], .01)
        self.assertEqual(model.body('ball').name, 'ball')
        for name in ('position', 'orientation', 'linear-velocity',
                     'angular-velocity', 'camera_position', 'camera_orientation'):
            self.assertGreaterEqual(model.sensor(name).id, 0)

        data = mujoco.MjData(model)
        q = np.linspace(-.15, .15, 13)
        data.qpos[:7] = [0, 0, .8, 1, 0, 0, 0]
        for i, name in enumerate(cfg['asset']['dof_names']):
            joint = model.joint(name).id
            data.qpos[model.jnt_qposadr[joint]] = q[i]
        mujoco.mj_forward(model, data)

        urdf = pin.buildModelFromUrdf(cfg['asset']['file'])
        frames = urdf.createData()
        pin.forwardKinematics(urdf, frames, q)
        pin.updateFramePlacements(urdf, frames)
        for name in cfg['asset']['foot_names']:
            expected = frames.oMf[urdf.getFrameId(name)].translation + [0, 0, .8]
            np.testing.assert_allclose(data.body(name).xpos, expected, atol=2e-6)
        mount = frames.oMf[urdf.getFrameId(cfg['asset']['head_name'])]
        camera_pos = mount.translation + mount.rotation @ cfg['asset']['camera_pos'] + [0, 0, .8]
        np.testing.assert_allclose(data.sensor('camera_position').data, camera_pos, atol=2e-6)
        mount_quat = pin.Quaternion(mount.rotation).coeffs()
        camera_quat = quat_mul(
            torch.tensor(mount_quat).unsqueeze(0),
            torch.tensor(cfg['asset']['camera_rot']).unsqueeze(0),
        )[0].numpy()
        actual_quat = data.sensor('camera_orientation').data[[1, 2, 3, 0]]
        self.assertAlmostEqual(abs(float(np.dot(actual_quat, camera_quat))), 1., places=6)

    def test_motion_schema_and_amp_sample(self):
        cfg = self.cfg
        names = cfg['asset']['dof_names']
        expected = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw',
                    'v_x', 'v_y', 'v_z', 'w_x', 'w_y', 'w_z']
        expected += ['q_' + n for n in names] + ['dq_' + n for n in names]
        paths = sorted(Path(cfg['motion']['motion_path']).glob('*.csv'))
        self.assertEqual(len(paths), 24)
        for path in paths:
            with path.open(newline='') as f:
                reader = csv.reader(f)
                self.assertEqual(next(reader), expected, str(path))
                self.assertTrue(all(len(row) == len(expected) for row in reader), str(path))
        motion = Motion(cfg, 'cpu')
        sample = motion.sample_batch(4)
        self.assertEqual(sample['dof_pos'].shape, (4, 13))
        current, following = next(motion.feed_forward_generator(1, 4))
        self.assertEqual(current.shape, (4, 41))
        self.assertEqual(following.shape, (4, 41))
        self.assertTrue(torch.isfinite(current).all())

    def test_reset_fk_matches_urdf(self):
        cfg = self.cfg
        frames = cfg['asset']['foot_names'] + [cfg['asset']['head_name']]
        fk = ResetKinematics(cfg['asset']['file'], cfg['asset']['dof_names'], frames, 'cpu')
        model = pin.buildModelFromUrdf(cfg['asset']['file'])
        q = torch.randn(3, 13) * .2
        result = fk(q)
        data = model.createData()
        for row in range(3):
            pin.forwardKinematics(model, data, q[row].numpy())
            pin.updateFramePlacements(model, data)
            for name in frames:
                placement = data.oMf[model.getFrameId(name)]
                np.testing.assert_allclose(result[name][0][row].numpy(), placement.translation, atol=2e-6)
                quat = pin.Quaternion(placement.rotation).coeffs()
                self.assertAlmostEqual(abs(float(np.dot(result[name][1][row].numpy(), quat))), 1., places=5)

    def test_mirror_and_action_scales_match_13_dofs(self):
        cfg = self.cfg
        scale = torch.tensor(cfg['control']['action_scale'])
        self.assertEqual(scale.shape, (13,))
        actions = torch.randn(8, 13)
        torch.testing.assert_close(G1.mirror_act(G1.mirror_act(actions)), actions)
        torch.testing.assert_close(G1.mirror_act(actions * scale), G1.mirror_act(actions) * scale)
        obs = torch.randn(8, cfg['env']['num_observations'])
        torch.testing.assert_close(G1.mirror_obs(G1.mirror_obs(obs)), obs)

    def test_mirrored_joints_reflect_foot_positions(self):
        cfg = self.cfg
        model = pin.buildModelFromUrdf(cfg['asset']['file'])
        data = model.createData()
        q = torch.randn(100, 13) * .25
        mirrored = G1.mirror_act(q)
        for original, reflected in zip(q.numpy(), mirrored.numpy()):
            positions = []
            for joints in (original, reflected):
                pin.forwardKinematics(model, data, joints)
                pin.updateFramePlacements(model, data)
                positions.append([
                    data.oMf[model.getFrameId(name)].translation.copy()
                    for name in cfg['asset']['foot_names']
                ])
            expected = np.asarray(positions[0])[[1, 0]] * [1, -1, 1]
            np.testing.assert_allclose(positions[1], expected, atol=2e-6)

    def test_amp_reference_matches_simulation_observation_layout(self):
        cfg = self.cfg
        motion = Motion(cfg, 'cpu')
        model = pin.buildModelFromUrdf(cfg['asset']['file'])
        data = model.createData()
        with Path(cfg['motion']['motion_path'], 'left_kick_1.csv').open(newline='') as f:
            reader = csv.reader(f)
            next(reader)
            rows = np.asarray([next(reader) for _ in range(4)], dtype=np.float32)
        for idx, row in enumerate(rows):
            quat = torch.from_numpy(row[3:7]).unsqueeze(0)
            base_pos = torch.from_numpy(row[:3]).unsqueeze(0)
            pin.forwardKinematics(model, data, row[13:26])
            pin.updateFramePlacements(model, data)
            foot_local = torch.as_tensor(np.stack([
                data.oMf[model.getFrameId(name)].translation.copy()
                for name in cfg['asset']['foot_names']
            ]), dtype=torch.float32)
            foot_world = base_pos + quat_rotate(quat.expand(2, -1), foot_local)
            env = G1.__new__(G1)
            env.cfg = cfg
            env.base_quat = quat
            env.base_pos = base_pos
            env.projected_gravity = quat_rotate_inverse(quat, torch.tensor([[0., 0., -1.]]))
            env.base_lin_vel = torch.from_numpy(row[7:10]).unsqueeze(0)
            env.base_ang_vel = torch.from_numpy(row[10:13]).unsqueeze(0)
            env.dof_pos = torch.from_numpy(row[13:26]).unsqueeze(0)
            env.dof_vel = torch.from_numpy(row[26:39]).unsqueeze(0)
            env.feet_pos = foot_world.unsqueeze(0)
            expected = motion.data[idx].clone()
            expected[22:35] *= cfg['normalization']['dof_vel']
            torch.testing.assert_close(env._compute_amp_obs()[0], expected, atol=2e-6, rtol=0)


if __name__ == '__main__':
    unittest.main()
