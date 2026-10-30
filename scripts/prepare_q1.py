"""Vendor the official Q1 URDF and generate a matching MuJoCo soccer scene.

Run from the repository root; requires numpy, scipy, pyyaml and mujoco.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def create_soccer_texture(path):
    """Seamless cube map of a spherical truncated icosahedron: 12 black pentagons."""
    from itertools import combinations
    from PIL import Image
    phi = (1 + math.sqrt(5)) / 2
    vertices = np.array([(0, a, b * phi) for a in (-1, 1) for b in (-1, 1)] +
                        [(a, b * phi, 0) for a in (-1, 1) for b in (-1, 1)] +
                        [(b * phi, 0, a) for a in (-1, 1) for b in (-1, 1)], dtype=float)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    distances = np.linalg.norm(vertices[:, None] - vertices[None, :], axis=-1)
    edges = np.isclose(distances, distances[distances > 0].min())
    normals, offsets = list(vertices), []
    for i, vertex in enumerate(vertices):
        j = np.flatnonzero(edges[i])[0]
        offsets.append(np.dot(vertex, (2 * vertex + vertices[j]) / 3))
    for i, j, k in combinations(range(12), 3):
        if edges[i, j] and edges[j, k] and edges[k, i]:
            normal = vertices[[i, j, k]].mean(0)
            normal /= np.linalg.norm(normal)
            normals.append(normal)
            offsets.append(np.dot(normal, (2 * vertices[i] + vertices[j]) / 3))
    normals, offsets = np.array(normals), np.array(offsets)
    size = 512
    u, v = np.meshgrid(2 * (np.arange(size) + .5) / size - 1, 2 * (np.arange(size) + .5) / size - 1)
    one = np.ones_like(u)
    directions = [(one, -v, -u), (-one, -v, u), (u, one, v),
                  (u, -one, -v), (u, -v, one), (-u, -v, -one)]
    images = []
    for xyz in directions:
        rays = np.stack(xyz, axis=-1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        scores = rays @ normals.T / offsets
        panel = scores.argmax(-1)
        top = np.partition(scores, -2, axis=-1)[..., -2:]
        seam = top[..., 1] - top[..., 0] < .006
        shade = np.where(panel < 12, .045, .91)
        shade = np.where(seam, .20, shade)
        grain = .008 * np.sin(211 * rays[..., 0] + 47 * rays[..., 2]) * np.sin(179 * rays[..., 1] - 31 * rays[..., 2])
        rgb = np.stack([shade + grain, shade + grain, shade + grain], axis=-1)
        images.append(np.uint8(np.clip(rgb, 0, 1) * 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(images, axis=0)).save(path)


def style_scene(scene):
    """Bright pitch and decorative goals; original collision geometry is retained."""
    world, asset = scene.find('worldbody'), scene.find('asset')
    for parent in (world, asset):
        for node in list(parent):
            if node.get('name', '').startswith('visual_') or node.tag == 'light':
                parent.remove(node)
    old_visual = scene.find('visual')
    if old_visual is not None:
        scene.remove(old_visual)
    visual = ET.SubElement(scene, 'visual')
    ET.SubElement(visual, 'headlight', ambient='0.4 0.4 0.4', diffuse='0.35 0.35 0.35', specular='0.1 0.1 0.1')
    ET.SubElement(visual, 'rgba', haze='0.78 0.86 0.94 1')
    ET.SubElement(visual, 'map', fogstart='25', fogend='60', znear='0.01')
    ET.SubElement(visual, 'quality', shadowsize='4096')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='720')
    ET.SubElement(asset, 'texture', name='visual_sky', type='skybox', builtin='gradient', width='512', height='3072',
                  rgb1='0.55 0.72 0.9', rgb2='0.92 0.96 0.99')
    ET.SubElement(asset, 'texture', name='visual_grass', type='2d', builtin='checker', width='256', height='256',
                  rgb1='0.15 0.35 0.17', rgb2='0.17 0.39 0.19')
    ET.SubElement(asset, 'material', name='visual_turf', texture='visual_grass', texrepeat='0.5 0.5',
                  texuniform='true', reflectance='0', specular='0.05', shininess='0.1')
    floor = world.find("geom[@name='floor']")
    floor.set('rgba', '1 1 1 1')
    floor.set('material', 'visual_turf')
    ET.SubElement(asset, 'texture', name='visual_football', type='cube', file='textures/soccer.png',
                  gridsize='6 1', gridlayout='RLUDFB')
    ET.SubElement(asset, 'material', name='visual_ball_material', texture='visual_football',
                  specular='0.25', shininess='0.35')
    world.find("body[@name='ball']/geom[@name='soccer_ball']").set('material', 'visual_ball_material')
    for collision in world.findall("body[@name='pelvis']/.//geom"):
        if collision.get('contype', '1') != '0':
            collision.set('group', '3')
    ET.SubElement(world, 'light', name='visual_sun', pos='-3 -4 8', dir='0.3 0.4 -1', directional='true',
                  ambient='0.1 0.1 0.1', diffuse='0.65 0.65 0.65', specular='0.15 0.15 0.15', castshadow='true')
    ET.SubElement(world, 'light', name='visual_fill', pos='4 5 6', dir='-0.4 -0.5 -1', directional='true',
                  diffuse='0.2 0.2 0.2', castshadow='false')

    def geom(name, **attrs):
        return ET.SubElement(world, 'geom', name='visual_' + name, contype='0', conaffinity='0',
                             group='1', density='0', rgba='0.96 0.97 0.95 1', **attrs)

    def rail(name, first, second, radius=.025):
        geom(name, type='capsule', fromto=' '.join(map(str, [*first, *second])), size=str(radius))

    def stripe(name, center, size):
        geom(name, type='box', pos=f'{center[0]} {center[1]} 0.007', size=f'{size[0]} {size[1]} 0.001')

    for x in (-7, 7):
        stripe(f'endline_{x}', (x, 0), (.018, 4.5))
    for y in (-4.5, 4.5):
        stripe(f'sideline_{y}', (0, y), (7, .018))
    # A thin, closed ring mesh avoids covering the grass inside the centre circle.
    vertices, faces = [], []
    for i in range(64):
        angle = 2 * math.pi * i / 64
        for radius in (.73, .76):
            vertices.extend([radius * math.cos(angle), radius * math.sin(angle), .008])
    for i in range(64):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * ((i + 1) % 64), 2 * ((i + 1) % 64) + 1
        faces.extend([a, b, c, b, d, c])
    ET.SubElement(asset, 'mesh', name='visual_center_ring', inertia='shell',
                  vertex=' '.join(map(str, vertices)), face=' '.join(map(str, faces)))
    geom('center_ring', type='mesh', mesh='visual_center_ring')

    # One lightweight mesh contains the net's back, sides and roof ribbons.
    vertices, faces = [], []
    def rectangle(origin, u, v):
        base = len(vertices) // 3
        for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
            vertices.extend([origin[k] + du * u[k] + dv * v[k] for k in range(3)])
        faces.extend([base, base + 1, base + 2, base, base + 2, base + 3,
                      base, base + 2, base + 1, base, base + 3, base + 2])

    def grid(origin, axis_u, length_u, axis_v, length_v):
        for axis, span, other, other_span in ((axis_u, length_u, axis_v, length_v), (axis_v, length_v, axis_u, length_u)):
            count = math.ceil(span / .13)
            for i in range(count + 1):
                point = list(origin)
                point[axis] += span * i / count - .003
                u, v = [0., 0., 0.], [0., 0., 0.]
                u[axis], v[other] = .006, other_span
                rectangle(point, u, v)

    grid((.55, -1.35, 0), 1, 2.7, 2, 1.05)
    for y in (-1.35, 1.35):
        grid((0, y, 0), 0, .55, 2, 1.05)
    grid((0, -1.35, 1.05), 0, .55, 1, 2.7)
    ET.SubElement(asset, 'mesh', name='visual_goal_net', inertia='shell',
                  vertex=' '.join(map(str, vertices)), face=' '.join(map(str, faces)))
    for sign, prefix in ((1, 'goal'), (-1, 'opponent_goal')):
        for suffix in ('left_post', 'right_post', 'back'):
            original = world.find(f"geom[@name='{prefix}_{suffix}']")
            original.set('rgba', '0 0 0 0')
            original.set('group', '4')
        front, back = sign * 7.05, sign * 7.6
        for y in (-1.35, 1.35):
            rail(f'{prefix}_front_{y}', (front, y, .045), (front, y, 1.05), .045)
            rail(f'{prefix}_rear_{y}', (back, y, .02), (back, y, 1.05), .018)
            rail(f'{prefix}_base_{y}', (front, y, .025), (back, y, .025), .018)
            rail(f'{prefix}_roof_{y}', (front, y, 1.05), (back, y, 1.05), .018)
        rail(f'{prefix}_crossbar', (front, -1.35, 1.05), (front, 1.35, 1.05), .045)
        rail(f'{prefix}_rear_crossbar', (back, -1.35, 1.05), (back, 1.35, 1.05), .018)
        net = geom(f'{prefix}_net', type='mesh', mesh='visual_goal_net', pos=f'{front} 0 0',
                   quat='1 0 0 0' if sign == 1 else '0 0 0 1')
        net.set('rgba', '0.88 0.94 0.98 0.7')
        for depth, half_width in ((1.2, 1.8), (2.2, 2.8)):
            stripe(f'{prefix}_area_front_{depth}', (sign * (7 - depth), 0), (.018, half_width))
            for y in (-half_width, half_width):
                stripe(f'{prefix}_area_side_{depth}_{y}', (sign * (7 - depth / 2), y), (depth / 2, .018))


def prepare(source):
    cfg = yaml.safe_load((ROOT / 'envs/Q1.yaml').read_text())
    dest = ROOT / 'resources/q1_description'
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / 'meshes', dest / 'meshes', dirs_exist_ok=True)
    (dest / 'urdf').mkdir(exist_ok=True)
    # Preserve vendor kinematics, collisions, inertias and joint limits verbatim.
    shutil.copy2(source / 'urdf/q1_v3.urdf', dest / 'urdf/q1_v3.urdf')
    urdf = ET.parse(dest / 'urdf/q1_v3.urdf').getroot()
    names = cfg['asset']['dof_names']
    joints = {j.get('name'): j for j in urdf.findall('joint') if j.get('type') != 'fixed'}
    assert list(joints) == names
    # Convert this exact URDF rather than the vendor's approximate MJCF transforms.
    model = mujoco.MjModel.from_xml_path(str(dest / 'urdf/q1_v3.urdf'))
    output = dest / 'q1_soccer.xml'
    mujoco.mj_saveLastXML(str(output), model)
    scene = ET.parse(output).getroot()
    scene.set('model', 'q1_soccer')
    compiler = scene.find('compiler')
    compiler.set('meshdir', 'meshes')
    for mesh in scene.findall('asset/mesh'):
        mesh.set('file', Path(mesh.get('file')).name)
    ET.SubElement(scene, 'option', timestep=str(cfg['sim']['dt']), gravity='0 0 -9.81')
    defaults = scene.find('default')
    if defaults is None:
        defaults = ET.SubElement(scene, 'default')
    ET.SubElement(defaults, 'joint', armature=str(cfg['asset']['armature']), damping='0')
    ET.SubElement(defaults, 'geom', friction='0.5 0.005 0.0001')
    world = scene.find('worldbody')
    pelvis = world.find("body[@name='pelvis']")
    if pelvis is None:
        raise ValueError('MuJoCo URDF importer did not preserve the pelvis')
    pelvis.set('pos', ' '.join(map(str, cfg['init_state']['pos'])))
    ET.SubElement(pelvis, 'freejoint', name='floating_base_joint')
    ET.SubElement(pelvis, 'site', name='policy_base', size='0.005')
    mount = world.find(f".//body[@name='{cfg['asset']['head_name']}']")
    q = cfg['asset']['camera_rot']
    ET.SubElement(mount, 'site', name='policy_camera', size='0.005',
                  pos=' '.join(map(str, cfg['asset']['camera_pos'])),
                  quat=' '.join(map(str, [q[3], *q[:3]])))
    for joint in world.findall('.//joint'):
        if joint.get('name') in joints:
            effort = joints[joint.get('name')].find('limit').get('effort')
            joint.set('actuatorfrcrange', f'-{effort} {effort}')
    # Isaac Gym's filter=1 disables robot self-collisions for this task.
    # Keep robot-field/ball contacts while making MuJoCo match that setting.
    for geom in pelvis.findall('.//geom'):
        if geom.get('contype', '1') != '0':
            geom.set('contype', '2')
            geom.set('conaffinity', '1')
    actuator = ET.SubElement(scene, 'actuator')
    for name in names:
        effort = joints[name].find('limit').get('effort')
        ET.SubElement(actuator, 'motor', name=name, joint=name, ctrllimited='true', ctrlrange=f'-{effort} {effort}')
    # Reuse the upstream field, goal and ball dimensions.
    field = ET.parse(ROOT / 'resources/g1_description/g1_soccer.xml').getroot()
    for node in field.find('worldbody'):
        world.append(node)
    for node in field.find('sensor'):
        sensor = scene.find('sensor')
        if sensor is None:
            sensor = ET.SubElement(scene, 'sensor')
        sensor.append(node)
    # Force sensors request MuJoCo's post-constraint inverse dynamics, making
    # cfrc_ext available for the batched collision reward.
    ET.SubElement(sensor, 'force', name='contact_force_trigger', site='policy_base')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='30 30 0.1', rgba='0.15 0.4 0.15 1')
    ET.SubElement(world, 'light', pos='0 0 5', dir='0 0 -1')
    create_soccer_texture(dest / 'textures/soccer.png')
    style_scene(scene)
    if hasattr(ET, 'indent'):
        ET.indent(scene)
    ET.ElementTree(scene).write(output, encoding='unicode')
    check = mujoco.MjModel.from_xml_path(str(output))
    assert [check.actuator(i).name for i in range(check.nu)] == names
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    (dest / 'SOURCE.json').write_text(json.dumps({
        'repository': 'https://github.com/PrimebotTech/primebot-q1-urdf',
        'commit': commit,
        'urdf_sha256': hashlib.sha256((dest / 'urdf/q1_v3.urdf').read_bytes()).hexdigest(),
        'mujoco_version': mujoco.__version__,
        'note': 'Unmodified official URDF and meshes; soccer MJCF generated from this URDF. Camera is a simulation assumption.',
    }, indent=2) + '\n')
    print(f'Prepared {check.nu}-DoF Q1 assets at {dest}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT.parent / 'primebot-q1-urdf')
    prepare(parser.parse_args().source.resolve())
