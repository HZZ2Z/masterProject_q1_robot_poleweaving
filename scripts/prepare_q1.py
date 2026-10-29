"""Vendor the official Q1 URDF and generate a matching MuJoCo soccer scene.

Run from the repository root; requires numpy, scipy, pyyaml and mujoco.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import yaml

ROOT = Path(__file__).resolve().parents[1]


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
