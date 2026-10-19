## 1st Home Assignment

This assignment will show you how to calibrate a projective camera, how to estimate the 3D position of known objects from their projections and how to render objects into a real scene (augmented reality).  When done, submit only the two *.py files into the information system.

We've prepared a set of calibration images, a video which you can use for augmentation testing, and a video with the expected output.

pov_du01_calibrate.py - Computes intrinsic camera parameters from a set of images containing a checkerboard - focal lengths, principal point and lens distortion parameters. The calibration is done using the function cv2.calibrateCamera(). **Exactly two lines of code contain errors.** Each error can be corrected by a small substitution within a single line of code. Do not add or delete lines of code.

pov_du01_augmented_reality.py - Localizes a checkerboard in 3D space using cv2.solvePnP(), places a point cloud (a simulated wave) into the checkerboard reference frame and projects those points onto the image. This script has two tasks:
1. Implement the projection yourself. Fill in the lines marked `# TODO: FILL THIS` in the function `project()`: the transformation from world to camera coordinates, the perspective division and the conversion to pixel coordinates. Lens distortion is already implemented. Do not use cv2.projectPoints() inside `project()`. You can use it (see the commented line in `main()`) to check that your implementation gives the same results.
2. One line of code in `main()` contains errors. Fix it without adding or deleting lines of code.

The document `projection_explained.pdf` explains the theory you need to fill in `project()`. It describes how a 3D point of the checkerboard world becomes a pixel in the image, in the same four steps as the numbered comments in `project()`:
1. world → camera coordinates (rotation and translation from cv2.solvePnP()),
2. perspective division onto the normalized image plane,
3. lens distortion (already implemented),
4. conversion to pixel coordinates using the camera matrix K.

For each step, the document gives the equation, a figure, and the corresponding lines of `pov_du01_augmented_reality.py` (with line numbers), so you can see exactly which equation belongs to which `# TODO` line.

The assignment is followed by a mandatory consultation, where you will explain how your code works.

Send any questions, comments or bug reports to Martin Kostelnik (ikostelnik@fit.vutbr.cz).


## MY EXPLANATION

#### pov_du01_calibrate.py: 

- **1st error:** `world_boards_points = [board_points for i in range(4)]`
  fixed to `world_boards_points = [board_points for i in range(len(image_points))]`.

  `cv2.calibrateCamera()` needs one set of 3D object points for every view in which
  the chessboard was detected. The code hard-coded 4 views, but the number of views
  depends on how many images were found and how many chessboards were detected
  (19 in my case). With `range(4)`, OpenCV received 4 sets of 3D points and 19 sets
  of 2D points, and failed with "nimages must be equal to imagePoints1.total()".
  Using `len(image_points)` makes the two lists always the same length, even if
  detection fails in some images.

- **2nd error:** `cv2.calibrateCamera(image_points, world_boards_points, img_shape, None, None)`
  fixed to `cv2.calibrateCamera(world_boards_points, image_points, img_shape, None, None)`.

  The function signature is `calibrateCamera(objectPoints, imagePoints, imageSize, ...)`:
  the 3D chessboard points in the world frame (`objectPoints`) go first, and the
  detected 2D corners in the image (`imagePoints`) go second. The two arguments
  were swapped, so OpenCV treated the 2D corners as 3D points and failed with
  "objectPoints should contain vector of vectors of points of type Point3f".
  Calibration works by finding the camera parameters that map the known 3D points
  to the observed 2D points, so the order matters.
