# Unitree G1 model

Robot description adapted from Unitree Robotics rev 1.0. The 13-DoF URDF for
Isaac Gym training and matching MJCF retain the 12 leg
joints and waist yaw. Waist roll and pitch and both arms are fixed at the
original training default pose.

Each foot uses one sole contact box over the former four corner points. This
keeps randomized contact materials below the PhysX 64K limit at 8192 environments.
