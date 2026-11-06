from pathlib import Path
import copy
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
COURSE_ROOT = Path(__file__).resolve().parents[1]
SCENE = COURSE_ROOT / 'resources/q1_description/q1_4v4.xml'
PREFIXES = [f'{team}{i}_' for team in ('red', 'blue') for i in range(4)]


def build_scene():
    tree = ET.parse(ROOT / 'resources/q1_description/q1_soccer.xml')
    root = tree.getroot()
    root.set('model', 'q1_4v4')
    world = root.find('worldbody')
    robot = world.find("body[@name='pelvis']")
    world.remove(robot)
    actuators = root.find('actuator')
    motors = list(actuators)
    actuators.clear()
    root.remove(root.find('sensor'))
    # Separate collision bit per robot: robot/robot and ball contacts enabled,
    # adjacent links within the same robot excluded as in the single-Q1 model.
    for i, prefix in enumerate(PREFIXES):
        body = copy.deepcopy(robot)
        sign = 1 if i < 4 else -1
        x, y = [(-5.8, 0), (-3.8, 0), (-1.8, -1.5), (-1.8, 1.5)][i % 4]
        body.set('pos', f'{sign*x} {sign*y} 0.417')
        body.set('quat', '1 0 0 0' if i < 4 else '0 0 0 1')
        for elem in body.iter():
            if 'name' in elem.attrib:
                elem.set('name', prefix + elem.get('name'))
            if elem.tag == 'geom' and elem.get('contype', '1') != '0':
                elem.set('contype', str(1 << (i + 1)))
                elem.set('conaffinity', str(511 ^ (1 << (i + 1))))
        head = body.find(f".//body[@name='{prefix}head_link']")
        color = '0.95 0.04 0.04 1' if i < 4 else '0.04 0.22 1 1'
        for name, size, pos in [('cap', '0.075 0.075 0.035', '0 0 0.09'),
                                ('brim', '0.10 0.085 0.008', '0.02 0 0.065')]:
            ET.SubElement(head, 'geom', name=prefix+name, type='ellipsoid',
                          size=size, pos=pos, rgba=color, contype='0',
                          conaffinity='0', mass='0', group='1')
        world.append(body)
        for motor in motors:
            m = copy.deepcopy(motor)
            m.set('name', prefix + m.get('name'))
            m.set('joint', prefix + m.get('joint'))
            actuators.append(m)
    ET.indent(tree)
    tree.write(SCENE, encoding='unicode')
    return SCENE


if __name__ == '__main__':
    print(build_scene())
