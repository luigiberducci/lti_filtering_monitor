import pandas as pd

import numpy as np
import sympy as sp

import signal_tl as stl
from signal_tl.monitors import lti_filter_monitor

T = 100

# define signals
SIGNALS = (
    'x_d', 'y_d', 'z_d',
    'roll', 'pitch', 'yaw',
    'dot_x', 'dot_y', 'dot_z',
    'dot_roll', 'dot_pitch', 'dot_yaw',
    'x_g', 'y_g', 'z_g',
)

(x_d, y_d, z_d, roll_d, pitch_d, yaw_d, dotx, doty, dotz, dotax, dotay, dotaz, x_g, y_g, z_g) = stl.signals(SIGNALS)

drone_pos = sp.Matrix([x_d, y_d, z_d])
drone_ori = sp.Matrix([roll_d, pitch_d, yaw_d])
drone_lv = sp.Matrix([dotx, doty, dotz])
drone_av = sp.Matrix([dotax, dotay, dotaz])

goal_pos = sp.Matrix([x_g, y_g, z_g])

# define specification
# reached goal => stay still
POSITION_SPEC = stl.Predicate((goal_pos - drone_pos).norm() <= 0.05)
GOAL_VELOCITY_SPEC = POSITION_SPEC >> stl.Predicate(drone_lv.norm() <= 0.001)

# Keep roll and pitch within 30deg
ANGLE_CONSTRAINT = (stl.Predicate(abs(roll_d) <= np.pi / 6) & stl.Predicate(abs(pitch_d) <= np.pi / 6))

# Minimize magnitude of angular velocity to <= 5deg/s
ANGULAR_VEL_CONSTRAINT = stl.Predicate(drone_av.norm() <= np.deg2rad(5))

SPEC = stl.G(  # Always do the following:
    stl.F(POSITION_SPEC)  # Head towards goal position
    & ANGLE_CONSTRAINT  # Keep the angles constrained
    & ANGULAR_VEL_CONSTRAINT  # COnstrain the angular velocity
    & GOAL_VELOCITY_SPEC  # Minimize drift once you reach goal
)

# define example trace
w = {
    "x_d": np.random.random((T,)),
    "y_d": np.random.random((T,)),
    "z_d": np.random.random((T,)),
    "yaw": np.random.random((T,)),
    "pitch": np.random.random((T,)),
    "roll": np.random.random((T,)),
    "dot_x": np.random.random((T,)),
    "dot_y": np.random.random((T,)),
    "dot_z": np.random.random((T,)),
    "dot_yaw": np.random.random((T,)),
    "dot_pitch": np.random.random((T,)),
    "dot_roll": np.random.random((T,)),
    "x_g": np.random.random((T,)),
    "y_g": np.random.random((T,)),
    "z_g": np.random.random((T,)),
}
w = pd.DataFrame(w, columns=SIGNALS)

# monitoring
monitor = lti_filter_monitor
result = monitor(phi=SPEC, w=w)
print(result)
