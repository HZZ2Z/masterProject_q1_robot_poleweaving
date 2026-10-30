import mujoco
import numpy as np
import torch
from q1_supermimic.physics import SCENE
from q1_supermimic.kinematics import Kinematics


def test_all_body_poses_and_velocities_match_native():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    fk = Kinematics(model)
    rng = np.random.default_rng(2107)
    qs, vs, positions, orientations, velocities = [], [], [], [], []
    for _ in range(8):
        data = mujoco.MjData(model)
        mujoco.mj_integratePos(model, data.qpos, rng.normal(size=model.nv), .3)
        data.qvel[:] = rng.normal(size=model.nv)
        mujoco.mj_forward(model, data)
        qs.append(data.qpos.copy()); vs.append(data.qvel.copy())
        positions.append(data.xpos.copy()); orientations.append(data.xquat.copy())
        velocity = []
        for body in range(model.nbody):
            # mj_objectVelocity(BODY) uses inertial-frame origin (COM),
            # whereas this interface explicitly returns body-frame origins.
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacobian, None, body)
            velocity.append(jacobian @ data.qvel)
        velocities.append(velocity)
    q, v = torch.tensor(np.array(qs)), torch.tensor(np.array(vs))
    p, rotation = fk(q)
    np.testing.assert_allclose(p.detach().numpy(), positions, atol=1e-10)
    actual = rotation.detach().numpy()
    target = np.array(orientations)
    signs = np.where((actual * target).sum(-1, keepdims=True) < 0, -1., 1.)
    np.testing.assert_allclose(actual * signs, target, atol=1e-10)
    linear, angular = fk.velocities(q, v)
    np.testing.assert_allclose(linear.detach().numpy(), velocities, atol=1e-9)
    for i in range(len(qs)):
        data = mujoco.MjData(model)
        data.qpos[:] = qs[i]; data.qvel[:] = vs[i]
        mujoco.mj_forward(model, data)
        for body in range(model.nbody):
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, None, jacobian, body)
            np.testing.assert_allclose(angular[i, body].detach().numpy(), jacobian @ data.qvel, atol=1e-9)


def test_foot_pose_and_velocity_have_finite_joint_gradients():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    fk = Kinematics(model)
    q = torch.tensor(model.qpos0, requires_grad=True)
    v = torch.ones(model.nv, dtype=torch.float64, requires_grad=True)
    foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_ankle_roll_link')
    position = fk(q)[0][foot]
    velocity = fk.linear_velocity(q, v)[foot]
    (position.square().sum() + velocity.square().sum()).backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(v.grad).all()
    assert q.grad[7:13].abs().sum() > 0
