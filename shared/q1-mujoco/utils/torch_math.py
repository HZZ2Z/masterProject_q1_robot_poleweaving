"""Simulator-independent quaternion helpers, using xyzw throughout."""
import torch


def quat_mul(a, b):
    av, bv = a[..., :3], b[..., :3]
    aw, bw = a[..., 3:], b[..., 3:]
    return torch.cat((aw * bv + bw * av + torch.cross(av, bv, dim=-1),
                      aw * bw - (av * bv).sum(-1, keepdim=True)), -1)


def quat_rotate(q, v):
    t = 2 * torch.cross(q[..., :3], v, dim=-1)
    return v + q[..., 3:] * t + torch.cross(q[..., :3], t, dim=-1)


def quat_rotate_inverse(q, v):
    return quat_rotate(torch.cat((-q[..., :3], q[..., 3:]), -1), v)


def quat_from_euler_xyz(r, p, y):
    sr, cr, sp, cp, sy, cy = (r / 2).sin(), (r / 2).cos(), (p / 2).sin(), (p / 2).cos(), (y / 2).sin(), (y / 2).cos()
    return torch.stack((sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                        cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy), -1)


def get_euler_xyz(q):
    x, y, z, w = q.unbind(-1)
    return (torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
            torch.asin((2*(w*y-z*x)).clamp(-1, 1)),
            torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def to_torch(value, device='cpu', dtype=torch.float):
    return torch.as_tensor(value, device=device, dtype=dtype)


def get_axis_params(value, axis):
    result = [0., 0., 0.]
    result[axis] = value
    return result


def torch_rand_float(lower, upper, shape, device='cpu'):
    return lower + (upper-lower) * torch.rand(shape, device=device)
