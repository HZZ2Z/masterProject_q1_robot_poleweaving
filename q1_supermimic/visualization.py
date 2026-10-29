"""Viewer-only Q1 optics overlay, copied from the PPO player for consistent display."""
import numpy as np
import mujoco


def quat_rotate(q, v):
    q_vec = q[:3]
    q_w = q[3]
    return v * (2.0 * q_w**2 - 1.0) + 2.0 * q_w * np.cross(q_vec, v) + 2.0 * q_vec * np.dot(q_vec, v)


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def camera_frustum_vertices(model, data, cfg, distance=2.5):
    """World-space optical frustum, using exactly the observation convention."""
    origin = data.sensor('camera_position').data.copy()
    camera = data.sensor('camera_orientation').data[[1, 2, 3, 0]]
    optical = quat_mul(camera, [.5] * 4)
    half_width, half_height = distance * np.tan(np.deg2rad(cfg['asset']['camera_fov']) / 2)
    corners = np.array([origin + quat_rotate(optical, np.array([x, y, distance]))
                        for x, y in ((-half_width, -half_height), (half_width, -half_height),
                                     (half_width, half_height), (-half_width, half_height))])
    return origin, corners


def draw_camera_frustum(scene, model, data, cfg, distance=2.5):
    """Viewer-only transparent faces; never contributes simulation geometry."""
    origin, corners = camera_frustum_vertices(model, data, cfg, distance)
    if scene.ngeom + 15 > scene.maxgeom:
        return

    def initialize(kind, size, pos, mat, rgba):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, kind, np.asarray(size, dtype=float), np.asarray(pos, dtype=float),
                           np.asarray(mat, dtype=float).reshape(9), np.asarray(rgba, dtype=np.float32))
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.emission = .45
        scene.ngeom += 1
        return geom

    def triangle(a, b, c, alpha):
        u, v = b - a, c - a
        normal = np.cross(u, v)
        mat = np.column_stack((u / np.linalg.norm(u), v / np.linalg.norm(v), normal / np.linalg.norm(normal)))
        initialize(mujoco.mjtGeom.mjGEOM_TRIANGLE, [np.linalg.norm(u), np.linalg.norm(v), 0], a, mat,
                   [.1, .7, 1., alpha])

    for i in range(4):
        triangle(origin, corners[i], corners[(i + 1) % 4], .075)
    triangle(corners[0], corners[1], corners[2], .035)
    triangle(corners[0], corners[2], corners[3], .035)
    for a, b in [(origin, corner) for corner in corners] + [(corners[i], corners[(i + 1) % 4]) for i in range(4)]:
        geom = initialize(mujoco.mjtGeom.mjGEOM_LINE, [0, 0, 0], [0, 0, 0], np.eye(3), [.15, .75, 1., .8])
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2., a, b)
    initialize(mujoco.mjtGeom.mjGEOM_SPHERE, [.012] * 3, origin, np.eye(3), [1., .65, .1, 1.])
