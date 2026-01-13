"""
Extended Kalman Filter with Acceleration Modeling

10-state model for tracking objects with acceleration:
State: [x, y, w, h, vx, vy, ax, ay, vw, vh]
- Position: (x, y) - top-left corner
- Size: (w, h) - width, height
- Velocity: (vx, vy) - pixels/frame
- Acceleration: (ax, ay) - pixels/frame^2
- Size rate: (vw, vh) - size change per frame
"""

import numpy as np
from typing import Tuple, Optional


class ExtendedKalmanTracker:
    """
    Extended Kalman Filter for bounding box tracking with acceleration modeling.

    Key improvement over constant-velocity Kalman:
    - Models acceleration explicitly
    - Predicts ahead for fast-moving objects
    - Acceleration decays over time (objects don't accelerate forever)
    """

    def __init__(self, bbox: Tuple[int, int, int, int],
                 dt: float = 1.0,
                 acceleration_decay: float = 0.9,
                 process_noise_pos: float = 0.5,
                 process_noise_vel: float = 1.0,
                 process_noise_acc: float = 2.0,
                 measurement_noise: float = 1.0):
        """
        Initialize EKF with initial bounding box.

        Args:
            bbox: Initial (x, y, w, h)
            dt: Time step (default 1 frame)
            acceleration_decay: How quickly acceleration decays (0.9 = slow decay)
            process_noise_pos: Process noise for position states
            process_noise_vel: Process noise for velocity states
            process_noise_acc: Process noise for acceleration states
            measurement_noise: Measurement noise
        """
        self.dt = dt
        self.acceleration_decay = acceleration_decay

        # State dimension: [x, y, w, h, vx, vy, ax, ay, vw, vh]
        self.n_states = 10
        self.n_measurements = 4

        # Store initial size for constraints
        x, y, w, h = bbox
        self.initial_w = w
        self.initial_h = h
        self.min_size_ratio = 0.5  # Don't shrink below 50% of original
        self.max_size_ratio = 1.5  # Don't grow above 150% of original

        # Initialize state
        self.state = np.array([
            x, y, w, h,  # Position and size
            0, 0,        # Velocity (vx, vy)
            0, 0,        # Acceleration (ax, ay)
            0, 0         # Size rate (vw, vh)
        ], dtype=np.float64)

        # State covariance
        self.P = np.eye(self.n_states, dtype=np.float64) * 10.0

        # Transition matrix (will be updated based on dt)
        self._update_transition_matrix()

        # Measurement matrix: we observe [x, y, w, h]
        self.H = np.zeros((self.n_measurements, self.n_states), dtype=np.float64)
        self.H[0, 0] = 1  # x
        self.H[1, 1] = 1  # y
        self.H[2, 2] = 1  # w
        self.H[3, 3] = 1  # h

        # Process noise covariance
        self.Q = np.eye(self.n_states, dtype=np.float64)
        self.Q[0:4, 0:4] *= process_noise_pos   # Position/size
        self.Q[4:6, 4:6] *= process_noise_vel   # Velocity
        self.Q[6:8, 6:8] *= process_noise_acc   # Acceleration
        self.Q[8:10, 8:10] *= process_noise_vel # Size rate

        # Measurement noise covariance
        self.R = np.eye(self.n_measurements, dtype=np.float64) * measurement_noise

        # For adaptive noise
        self.base_Q = self.Q.copy()
        self.innovation_history = []

    def _update_transition_matrix(self):
        """Update transition matrix based on dt."""
        dt = self.dt
        dt2 = 0.5 * dt * dt
        decay = self.acceleration_decay

        # State transition: x' = x + vx*dt + 0.5*ax*dt^2
        #                   vx' = vx + ax*dt
        #                   ax' = ax * decay
        self.F = np.array([
            # x     y     w     h    vx    vy    ax     ay    vw    vh
            [1,    0,    0,    0,   dt,    0,   dt2,    0,    0,    0],   # x
            [0,    1,    0,    0,    0,   dt,    0,   dt2,    0,    0],   # y
            [0,    0,    1,    0,    0,    0,    0,     0,   dt,    0],   # w
            [0,    0,    0,    1,    0,    0,    0,     0,    0,   dt],   # h
            [0,    0,    0,    0,    1,    0,   dt,     0,    0,    0],   # vx
            [0,    0,    0,    0,    0,    1,    0,    dt,    0,    0],   # vy
            [0,    0,    0,    0,    0,    0, decay,    0,    0,    0],   # ax (decays)
            [0,    0,    0,    0,    0,    0,    0, decay,    0,    0],   # ay (decays)
            [0,    0,    0,    0,    0,    0,    0,     0,    1,    0],   # vw
            [0,    0,    0,    0,    0,    0,    0,     0,    0,    1],   # vh
        ], dtype=np.float64)

    def predict(self, dt: Optional[float] = None) -> Tuple[int, int, int, int]:
        """
        Predict next state.

        Args:
            dt: Time step (uses default if None)

        Returns:
            Predicted bbox (x, y, w, h)
        """
        if dt is not None and dt != self.dt:
            self.dt = dt
            self._update_transition_matrix()

        # State prediction: x' = F * x
        self.state = self.F @ self.state

        # Covariance prediction: P' = F * P * F' + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.get_bbox()

    def update(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Update state with measurement.

        Args:
            bbox: Measured (x, y, w, h)

        Returns:
            Updated bbox estimate
        """
        z = np.array([bbox[0], bbox[1], bbox[2], bbox[3]], dtype=np.float64)

        # Innovation: y = z - H * x
        y = z - self.H @ self.state

        # Store innovation for adaptive noise
        self.innovation_history.append(np.linalg.norm(y))
        if len(self.innovation_history) > 30:
            self.innovation_history.pop(0)

        # Innovation covariance: S = H * P * H' + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: K = P * H' * S^-1
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fallback if S is singular
            K = self.P @ self.H.T @ np.linalg.pinv(S)

        # State update: x = x + K * y
        self.state = self.state + K @ y

        # Covariance update: P = (I - K*H) * P
        I = np.eye(self.n_states)
        self.P = (I - K @ self.H) @ self.P

        return self.get_bbox()

    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Get current bounding box estimate with size constraints."""
        x, y, w, h = self.state[0:4]

        # Constrain size to reasonable bounds relative to initial size
        min_w = self.initial_w * self.min_size_ratio
        max_w = self.initial_w * self.max_size_ratio
        min_h = self.initial_h * self.min_size_ratio
        max_h = self.initial_h * self.max_size_ratio

        w = max(min_w, min(max_w, w))
        h = max(min_h, min(max_h, h))

        # Also update state to prevent drift
        self.state[2] = w
        self.state[3] = h

        return (int(x), int(y), int(w), int(h))

    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate (vx, vy)."""
        return (float(self.state[4]), float(self.state[5]))

    def get_acceleration(self) -> Tuple[float, float]:
        """Get current acceleration estimate (ax, ay)."""
        return (float(self.state[6]), float(self.state[7]))

    def get_speed(self) -> float:
        """Get speed magnitude."""
        vx, vy = self.get_velocity()
        return float(np.sqrt(vx**2 + vy**2))

    def get_acceleration_magnitude(self) -> float:
        """Get acceleration magnitude."""
        ax, ay = self.get_acceleration()
        return float(np.sqrt(ax**2 + ay**2))

    def predict_future(self, n_frames: int = 1) -> Tuple[int, int, int, int]:
        """
        Predict position n frames ahead WITHOUT updating internal state.
        Useful for computing search regions.

        Args:
            n_frames: Number of frames to predict ahead

        Returns:
            Predicted bbox
        """
        # Temporarily predict ahead
        x, y, w, h = self.state[0:4]
        vx, vy = self.state[4:6]
        ax, ay = self.state[6:8]

        t = n_frames * self.dt

        # x = x0 + v0*t + 0.5*a*t^2
        x_future = x + vx * t + 0.5 * ax * t * t
        y_future = y + vy * t + 0.5 * ay * t * t

        return (int(x_future), int(y_future), int(w), int(h))

    def get_position_uncertainty(self) -> Tuple[float, float]:
        """
        Get position uncertainty (standard deviation).

        Returns:
            (sigma_x, sigma_y)
        """
        return (np.sqrt(self.P[0, 0]), np.sqrt(self.P[1, 1]))

    def set_acceleration(self, ax: float, ay: float):
        """
        Manually set acceleration (useful when external motion is detected).

        Args:
            ax: X acceleration
            ay: Y acceleration
        """
        self.state[6] = ax
        self.state[7] = ay

    def boost_process_noise(self, factor: float = 2.0):
        """
        Temporarily increase process noise (useful during high uncertainty).

        Args:
            factor: Multiplier for process noise
        """
        self.Q = self.base_Q * factor

    def reset_process_noise(self):
        """Reset process noise to base level."""
        self.Q = self.base_Q.copy()

    def adapt_noise(self, phase: str = "normal"):
        """
        Adapt process noise based on tracking phase.

        Args:
            phase: "stationary", "camera_following", or "drone_accelerating"
        """
        if phase == "stationary":
            # Low noise - object not moving much
            self.Q = self.base_Q * 0.5
        elif phase == "camera_following":
            # Medium noise - camera is moving, object relatively still in frame
            self.Q = self.base_Q * 1.0
        elif phase == "drone_accelerating":
            # High noise - object accelerating, need to track rapid changes
            self.Q = self.base_Q * 3.0
            # Extra boost for acceleration states
            self.Q[6:8, 6:8] *= 2.0
        else:
            self.Q = self.base_Q.copy()


def center_to_tlwh(cx: float, cy: float, w: float, h: float) -> Tuple[int, int, int, int]:
    """Convert center coordinates to top-left + width/height."""
    return (int(cx - w/2), int(cy - h/2), int(w), int(h))


def tlwh_to_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
    """Convert top-left + width/height to center coordinates."""
    x, y, w, h = bbox
    return (x + w/2, y + h/2, w, h)
