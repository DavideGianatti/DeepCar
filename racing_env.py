import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

dt = 0.1                  # time step (s)
friction = 1              # friction coefficient between the wheels and the road
crash_reward = - 100 * dt

class Car:
    """
    Class describing a car moving in a 2D space: it can accelerate, decelerate, turn left/right and crash
    """

    def __init__(self, pos, v, track, max_acc=5, min_dec=5, min_r=10, progress=0):
        """
        Initializes a new Car instance with the given parameters.

        Args:
            pos (tuple): Position of the car in a 2D space represented as (x, y) in meters.
            v (tuple): Velocity of the car represented as (vx, vy) in meters per second.
            track (Track): An instance of the Track class representing the racetrack on which the car is driving.
            max_acc (float): Maximum possible acceleration of the car, in meters per second squared (m/s^2). Default is 5.
            min_dec (float): Maximum possible deceleration of the car, in meters per second squared (m/s^2). Default is 5.
            min_r (float): Minimum turning radius of the car, in meters (m). Default is 10.
            progress (int): Index of the current racetrack segment where the car is positioned. Default is 0.
        """

        self.pos = np.array(pos)
        self.v = np.array(v)
        self.dir = self.v / np.sqrt(np.sum(self.v ** 2))
        self.max_acc = max_acc
        self.min_dec = min_dec
        self.min_r = min_r
        self.progress = self.get_progress(progress, track)
        if np.dot(self.dir, track.dir[self.progress]) < 0:
            raise ValueError('Car going in the wrong direction!')

    def move(self, left_right, gas_break, track):

        """
        Moves the car by updating its position and velocity based on acceleration, deceleration, and turning.

        Args:
            left_right (float): Turning input; reciprocal of the turning radius. Negative for left turns, positive for right turns.
            gas_break (float): Fraction of the maximum acceleration (positive) or deceleration (negative) to apply.
            track (Track): The track object on which the car is moving.

        Returns:
            float: The reward for the movement.
        """

        init_pos = np.copy(self.pos)
        init_progress = np.copy(self.progress)

        if gas_break > 0:
            self.v += gas_break * self.max_acc * dt * self.dir

        else:
            if abs(gas_break * self.min_dec * dt) >= np.linalg.norm(self.v):
                self.v = self.dir               # avoid divergences, |v| = 3.6 km/h
            else:
                self.v += gas_break * self.min_dec * dt * self.dir


        # minimum turning radius allowed by the friction
        min_tr = np.sum(self.v ** 2) / 9.81 * friction
        min_tr = max(min_tr, self.min_r)

        turning_radius = 1 / (left_right + 10**(-6))     # add epsilon to avoid divergences
        turning_radius = max(abs(turning_radius), min_tr) * np.sign(turning_radius)

        # c: center of the turning circumference
        c = self.pos + turning_radius * perpendicular(self.dir)
        self.pos, self.v = circular_motion(c=c, r=abs(turning_radius), v=self.v,
                                           left_right=np.sign(turning_radius))

        self.dir = self.v / np.linalg.norm(self.v)

        self.progress = self.get_progress(self.progress, track)

        reward = self.get_reward(track, init_pos)

        if gas_break < 0:
            reward = - dt

        if self.crash(track, init_pos, init_progress):
            reward = crash_reward

        self.check_bug(track)

        return reward

    def get_progress(self, progress, track):
        """
        Determines the index of the nearest track discretization point before the car's current position.

        Args:
            progress (int): Current progress index on the track.
            track (Track): The track object on which the car is moving.

        Returns:
            int: Updated progress index on the track.

        Raises:
            ValueError: If the progress index cannot be determined.
        """
        next_ind = (progress + 1) % track.n_segment

        count = 0
        while (np.linalg.norm(self.pos - track.center[progress]) > np.linalg.norm(self.pos - track.center[next_ind])) \
                and (np.dot(track.center[next_ind] - self.pos, track.dir[next_ind]) < 0):
            progress += 1
            progress %= track.n_segment

            count += 1
            if count > track.n_segment:
                raise ValueError('Progress not found!')

        return progress

    def crash(self, track, init_pos, init_progress):
        """
        Checks if the car has crashed into the track boundaries.

        Args:
            track (Track): The track object on which the car is moving.
            init_pos (np.array): Car's position before the last time step.
            init_progress (int): Progress index before the last time step.

        Returns:
            bool: True if the car has crashed into a hedge, otherwise False.
        """

        # avoid divergences
        if init_pos[0] == self.pos[0]:
            init_pos[0] += 0.1

        # linear coefficient describing the linear trajectory of the car
        linear_trajectory = linear_fit(init_pos, self.pos)

        # check for crash in every segment passed
        progress = init_progress
        next_progress = (progress + 1) % track.n_segment
        while True:

            # solve linear systems with hedges to check if the car crashed
            x = (track.linear_hedge[:, progress, 1] - linear_trajectory[1]) / \
                (linear_trajectory[0] - track.linear_hedge[:, progress, 0])
            y = x * linear_trajectory[0] + linear_trajectory[1]

            # left
            if is_between(x[0], init_pos[0], self.pos[0]) and \
               is_between(x[0], track.hedge[0, progress, 0], track.hedge[0, next_progress, 0]):
                self.pos[0] = x[0]
                self.pos[1] = y[0]
                self.progress = progress
                self.dir = track.hedge[0, (progress + 1) % track.n_segment] - track.hedge[0, progress]
                self.dir /= np.linalg.norm(self.dir)
                self.pos += perpendicular(self.dir)   # set car 1m inside the track
                self.v = self.dir * 0.01              # avoid divergences (v != 0)

                return True

            # right
            if is_between(x[1], init_pos[0], self.pos[0]) and \
               is_between(x[1], track.hedge[1, progress, 0], track.hedge[1, next_progress, 0]):
                self.pos[0] = x[1]
                self.pos[1] = y[1]
                self.progress = progress
                self.dir = track.hedge[1, (progress + 1) % track.n_segment] - track.hedge[1, progress]
                self.dir /= np.linalg.norm(self.dir)
                self.pos -= perpendicular(self.dir)  # set car 1m inside the track
                self.v = self.dir * 0.01        # avoid divergences (v != 0)

                return True

            if progress == self.progress:
                break

            progress = next_progress
            next_progress = (progress + 1) % track.n_segment

        return False

    def check_bug(self, track):
        """
        Ensures that the car is still within the track boundaries.

        Args:
            track (Track): The track object on which the car is moving.

        Raises:
            ValueError: If the car is outside the track boundaries.
        """

        if np.linalg.norm(self.pos - track.center[self.progress]) > track.max_lenght:
            raise ValueError('Car is outside the track!')

    def get_reward(self, track, init_pos):
        """
        Calculates the reward based on the car's position relative to the track center.

        Args:
            track (Track): The track object on which the car is moving.
            init_pos (np.array): Car's position before the last time step.

        Returns:
            float: Reward value, based on the distance from the track center.
        """
        road_dir = track.dir[self.progress]
        aux = linear_fit(init_pos, init_pos + perpendicular(road_dir))
        m_road = aux[0]
        b_road = aux[1]

        reward = d_point_line(m_road, b_road, self.pos[0], self.pos[1]) / dt

        return reward


class Track:
    """
    Class defining a racing track in a 2D space for cars
    """

    def __init__(self, center, width):
        """
        Initializes the Track with the given parameters.

        Args:
            center (list): A list of (x, y) coordinates representing the center points of the track segments.
            width (float): The width of the track.

        Attributes:
            center (np.array): Array of center points of the track segments.
            n_segment (int): Number of discretization points (segments) of the track.
            linear_center (np.array): Linear interpolation coefficients (m, b) of the track center line.
            width (float): Width of the track.
            hedge (np.array): Track hedges' coordinates (left/right).
            linear_hedge (np.array): Linear interpolation coefficients (m, b) for the hedges.
            dir (np.array): Piece-wise direction versors of the track segments.
            max_lenght (float): Maximum length of the track's discretization segment.
        """

        self.center = np.array(center)
        self.n_segment = len(self.center)
        self.linear_center = self.get_linear_center()
        self.width = width
        self.dir, self.max_lenght = self.get_dir()
        self.max_lenght = np.max(self.max_lenght)
        self.hedge = self.get_hedge()
        self.linear_hedge = self.get_linear_hedge()

    def __call__(self, car, angles):
        """
        Computes the distances from the car to the track's hedges at various angles.

        Args:
            car (Car): A car object.
            angles (np.array): List of angles (in radians) representing the car's point of view.

        Returns:
            np.array: Distances from the car to the track's hedges at the specified angles, in meters.
        """

        # linear coefficients (m, b) describing the direction of the car
        linear_car = linear_fit(car.pos, car.pos + car.dir)

        angles = np.array(angles)
        n_angles = len(angles)

        # distances of the track's hedges from the car
        distances = np.zeros(n_angles)

        # starting segment is n_previous-points before the car's nearest
        n_previous = 1

        for i in range(n_angles):

            ang = angles[i]

            ind = car.progress - n_previous

            # point of view to be checked, expressed in linear coefficients (m, b)
            pov = np.zeros(2)
            pov[0] = np.tan(np.arctan(linear_car[0]) + ang)
            pov[1] = car.pos[1] - pov[0] * car.pos[0]

            count = 0
            while True:

                d = self.hedge_on_the_way(ind, car, pov, left_right=1)
                if d:
                    break

                d = self.hedge_on_the_way(ind, car, pov, left_right=0)
                if d:
                    break

                ind += 1
                if ind > 0:
                    ind %= self.n_segment

                count += 1
                if count > self.n_segment:
                    raise ValueError('Distance from hedge not found!')

            distances[i] = d

        return distances / 100

    def hedge_on_the_way(self, ind, car, pov, left_right):
        """
        Computes the distance from the car to the hedge of the track at a given segment and point of view.

        Args:
            ind (int): Index of the track's segment to be checked.
            car (Car): The car object.
            pov (np.array): Linear coefficients (m, b) describing the car's point of view.
            left_right (int): 0 for left hedge, 1 for right hedge.

        Returns:
            float: Distance from the car to the hedge. Returns 0 if no interception.
        """

        x = (self.linear_hedge[left_right, ind, 1] - pov[1]) / (pov[0] - self.linear_hedge[left_right, ind, 0])
        y = x * self.linear_hedge[left_right, ind, 0] + self.linear_hedge[left_right, ind, 1]
        r = np.array([x, y]) - car.pos
        d = 0
        if is_between(x, self.hedge[left_right, ind, 0], self.hedge[left_right, (ind + 1) % self.n_segment, 0]) and \
           is_between(y, self.hedge[left_right, ind, 1], self.hedge[left_right, (ind + 1) % self.n_segment, 1]) and \
           round(np.dot(r, car.dir), 2) >= 0:
            d = np.linalg.norm(r)

        return d


    def get_linear_center(self):
        """
        Computes the linear coefficients (m, b) for the track's center line.

        Returns:
            np.array: Array of linear coefficients (m, b) for each track segment.
        """

        linear_center = np.zeros((self.n_segment, 2))

        for i in range(self.n_segment):
            next_index = (i + 1) % self.n_segment
            linear_center[i] = linear_fit(self.center[i], self.center[next_index])

        return linear_center

    def get_dir(self):
        """
        Computes the direction vectors and lengths of the track segments.

        Returns:
            tuple:
                - np.array: Direction vectors for each track segment.
                - np.array: Lengths of the track segments.
        """

        direction = np.zeros((self.n_segment, 2))
        lenght_segment = np.zeros(self.n_segment)

        for i in range(self.n_segment):
            next_index = (i + 1) % self.n_segment
            direction[i] = self.center[next_index] - self.center[i - 1]
            lenght_segment[i] = np.linalg.norm(direction[i])
            direction[i] /= lenght_segment[i]

        return direction, lenght_segment

    def get_hedge(self):
        """
        Computes the coordinates of the track's hedges' discretization points (left/right).

        Returns:
            np.array: Array containing the coordinates of the left and right hedges for each segment.
        """

        hedge = np.zeros((2, self.n_segment, 2))
        for i in range(self.n_segment):
            hedge[0, i] = self.center[i] - perpendicular(self.dir[i]) * self.width
            hedge[1, i] = self.center[i] + perpendicular(self.dir[i]) * self.width

        return hedge

    def get_linear_hedge(self):
        """
        Computes the linear coefficients (m, b) for the track's hedges.

        Returns:
            np.array: Array of linear coefficients (m, b) for each hedge of the track.
        """

        linear_hedge = np.zeros((2, self.n_segment, 2))

        for i in range(self.n_segment):
            next_index = (i + 1) % self.n_segment

            linear_hedge[0, i] = linear_fit(self.hedge[0, i], self.hedge[0, next_index])
            linear_hedge[1, i] = linear_fit(self.hedge[1, i], self.hedge[1, next_index])

        return linear_hedge


def is_between(x, x0, x1):
    """
    Checks if a value x is between two other values x0 and x1.

    Args:
        x (float): Value to check.
        x0 (float): Lower bound.
        x1 (float): Upper bound.

    Returns:
        bool: True if x is between x0 and x1, False otherwise.
    """

    x = round(x, 2)         # precision of 1 cm
    x0 = round(x0, 2)
    x1 = round(x1, 2)

    x_max = max(x0, x1)
    x_min = min(x0, x1)

    return x_min <= x <= x_max



def linear_fit(r1, r2):
    """
    Computes the linear equation coefficients for a line passing through two points.

    Args:
        r1 (np.array): First point (x, y).
        r2 (np.array): Second point (x, y).

    Returns:
        np.array: Linear coefficients (m, b) where y = m * x + b.
    """

    m = (r2[1] - r1[1]) / (r2[0] - r1[0] + 0.0001)  # add 0.1 mm in order to avoid divergences
    b = r1[1] - m * r1[0]

    return np.array([m, b])


def circular_motion(c, r, v, left_right):
    """
    Computes the new position and velocity of a point moving in a circular path.

    Args:
        c (tuple): Center of the circle (x, y).
        r (float): Radius of the circle.
        v (np.array): Initial velocity vector (vx, vy).
        left_right (int): Direction of rotation (1 for clockwise, -1 for counterclockwise).

    Returns:
        tuple: New position (x, y) and new velocity vector (vx, vy).
    """

    phi = np.arctan2(left_right * v[0], - left_right * v[1])
    v_magnitude = np.linalg.norm(v)
    omega = - left_right * v_magnitude / r
    phi += omega * dt

    pos = c + np.array([r * np.cos(phi), r * np.sin(phi)])
    v = np.array([v_magnitude * np.sin(phi), - v_magnitude * np.cos(phi)]) * left_right

    return pos, v


def perpendicular(x):
    """
    Computes the vector perpendicular to the given vector, rotated by -pi/2 radians.

    Args:
        x (np.array): Input vector (vx, vy).

    Returns:
        np.array: Perpendicular vector.
    """

    return np.array([x[1], -x[0]])


def d_point_line(m, b, x, y):
    """
    Computes the distance from a point to a straight line given by its linear equation.

    Args:
        m (float): Slope of the line.
        b (float): Intercept of the line.
        x (float): x-coordinate of the point.
        y (float): y-coordinate of the point.

    Returns:
        float: Distance from the point to the line.
    """

    return np.absolute(y - m * x - b) / np.sqrt(1 + m ** 2)
