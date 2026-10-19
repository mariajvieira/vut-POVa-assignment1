# https://longervision.github.io/2017/03/12/ComputerVision/OpenCV/opencv-external-posture-estimation-ArUco-board/

import numpy as np
import cv2
import argparse


def parseargs():
    parser = argparse.ArgumentParser('Augmented reality on a checkerboard.')
    parser.add_argument('-v', '--video-file', help='Process a video file. If not specified, opens first camera.')
    parser.add_argument('-c', '--camera-params', required=True, help='File from calibrate.py.')
    parser.add_argument('--chessboard-rows', default=6, type=int, help='Number of chessboard inner corners.')
    parser.add_argument('--chessboard-cols', default=4, type=int, help='Number of chessboard inner corners.')
    args = parser.parse_args()
    return args


def rodrigues(rvec):
    """Axis-angle vector -> 3x3 rotation matrix."""
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def project(points, K, rvec, tvec, dist):
    """Project Nx3 world points to Nx1x2 pixel coordinates (OpenCV camera model, same output format as cv2.projectPoints)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    dist = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64).ravel()
    dist = np.pad(dist, (0, max(0, 5 - dist.size)))
    k1, k2, p1, p2, k3 = dist[:5]

    # 1. World -> camera coordinates
    R = rodrigues(rvec)

    # The formula X_c = R * X_w + t transforms ONE point written as a column vector.
    # Here, `points` stores one point per ROW (shape (N, 3)), so we use the transposed
    # version of the formula: (R * X_w)^T = X_w^T * R^T. 
    # This lets us rotate all N points with a single matrix product (points @ R.T), 
    # and then add the translation t, reshaped from (3, 1) to (1, 3) so that NumPy broadcasting adds it to every row. 
    # The result, X_cam, has shape (N, 3) and holds each point in the camera frame (origin at the
    # camera centre, Z_c = depth along the optical axis).


    # X_c = R * X_w + t   ==>    X_c(row) = X_w(row) * Rᵀ + tᵀ
    # R.T is the transpose of the matrix
    X_cam = points @ R.T + tvec.reshape(1,3)   # TODO: FILL THIS

    # 2. Perspective division -> normalized image plane (z = 1)
    # x = Xc/Zc, y = Yc/Zc
    x = X_cam[:, 0] / X_cam[:, 2]       # TODO: FILL THIS
    y = X_cam[:, 1] / X_cam[:, 2]       # TODO: FILL THIS

    # 3. Lens distortion (radial k1, k2, k3 + tangential p1, p2)
    r2 = x**2 + y**2
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    x_d = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    y_d = y * radial + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y


    # 4. Intrinsics: normalized -> pixel coordinates
    # u = f_x * x_d + s * y_d + c_x * 1 
    u = K[0, 0] * x_d + K[0, 1] * y_d + K[0, 2]  # TODO: FILL THIS K[0, 0] * x_d + K[0, 1] * y_d + K[0, 2] 
    # v = 0 * x_d + f_y * y_d + c_y * 1
    v = K[1, 0] * x_d + K[1, 1] * y_d + K[1, 2]  # TODO: FILL THIS K[1, 0] * x_d + K[1, 1] * y_d + K[1, 2]

    return np.stack([u, v], axis=1).reshape(-1, 1, 2)


# Wave simulation. This is not part of the assignment and there are no errors here.
# from https://beltoforion.de/en/recreational_mathematics/2d-wave-equation.php
class WaveSim:
    def __init__(self, width=64, height=64):
        self.u = np.zeros((3, height, width))         # The three dimensional simulation grid
        self.c = 0.5                                # The "original" wave propagation speed
        self.h = 1  # spatial step width
        self.k = 1  # time step width
        self.alpha = np.zeros((height, width))        # wave propagation velocities of the entire simulation domain
        self.alpha[...] = ((self.c*self.k) / self.h)**2  # will be set to a constant value of tau

        y = np.random.randint(5, height - 5)
        x = np.random.randint(5, width - 5)
        self.u[0, x - 2:x + 2, y - 2:y + 2] = 100

    def update(self):
        self.u[2] = self.u[1]
        self.u[1] = self.u[0]
        s1 = self.u.shape[1]
        s2 = self.u.shape[2]

        self.u[0, 1:s1-1, 1:s2-1] = self.alpha[1:s1-1, 1:s2-1] * (self.u[1, 0:s1-2, 1:s2-1] +
                                        self.u[1, 2:s1,   1:s2-1] +
                                        self.u[1, 1:s1-1, 0:s2-2] +
                                        self.u[1, 1:s1-1, 2:s2] - 4*self.u[1, 1:s1-1, 1:s2-1]) \
                                        + 2 * self.u[1, 1:s1-1, 1:s2-1] - self.u[2, 1:s1-1, 1:s2-1]

        # Not part of the wave equation but I need to remove energy from the system.
        # The boundary conditions are closed. Energy cannot leave and the simulation keeps adding energy.
        self.u[0, 1:s1-1, 1:s2-1] *= 0.99

        if np.random.uniform() < 0.02:
            y = np.random.randint(5, s1-5)
            x = np.random.randint(5, s2-5)
            self.u[0, x-2:x+2, y-2:y+2] = 120

    def get_points(self, width=5, height=5):
        x = np.linspace(0, width, num=self.u.shape[2] - 2)
        y = np.linspace(0, height, num=self.u.shape[1] - 2)
        xv, yv = np.meshgrid(x, y, indexing='ij')
        coords = np.stack([xv, yv, self.u[0, 1:-1, 1:-1]/400 - 1], axis=2)
        return coords.reshape(-1, 3)


# Just draw points 2D, nothing special here and no errors
def draw_points(img, points, colors, color_low=np.asarray([255, 0, 0]), color_high=np.asarray([0, 0, 255]),  point_size=2):
    points = points.astype(int)
    for i, p in enumerate(points[:, 0]):
        color = colors[i] * color_low + (1-colors[i]) * color_high
        color = tuple(color.astype(np.uint8).tolist())
        cv2.circle(img, (p[0], p[1]), radius=point_size, color=color, thickness=-1)


def main():
    args = parseargs()

    # the wave simulation
    wave_sim = WaveSim(64, 64)

    # The size of chessboard
    chessboard_rows = args.chessboard_rows
    chessboard_cols = args.chessboard_cols

    npzfile = np.load(args.camera_params)
    mtx = npzfile['mtx']
    dist = npzfile['dist']

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001)

    board_points = np.zeros((chessboard_cols * chessboard_rows, 3), np.float32)
    board_points[:, :2] = np.mgrid[0:chessboard_rows, 0:chessboard_cols].T.reshape(-1, 2)

    if args.video_file:
        capture = cv2.VideoCapture(args.video_file)
    else:
        capture = cv2.VideoCapture()

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        wave_sim.update()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.blur(frame_gray, (5, 5))
        ret, corners = cv2.findChessboardCorners(frame_gray, (chessboard_rows, chessboard_cols), None)
        if ret:
            corners = cv2.cornerSubPix(frame_gray, corners, (7, 7), (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, (chessboard_rows, chessboard_cols), corners, ret)

            # Find the rotation and translation vectors.
            # OLD LINE: ret, rvecs, tvecs = cv2.solvePnP(corners, board_points, mtx, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            ret, rvecs, tvecs = cv2.solvePnP(board_points, corners, mtx, dist)


            # project 3D points to image plane
            points = wave_sim.get_points()
            # points_img, _ = cv2.projectPoints(points, rvecs, tvecs, mtx, dist) # Use this if you want to use OpenCV's projectPoints function for comparison
            points_img = project(points, mtx, rvecs, tvecs, dist)

            # get point colors from z coordinate
            colors = np.clip((points[:, 2] + 1) * 2 + 0.5, 0, 1)

            draw_points(frame, points_img, colors)

        cv2.imshow('img', frame)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('x'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
