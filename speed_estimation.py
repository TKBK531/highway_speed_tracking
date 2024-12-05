from time import time
import numpy as np
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import uuid


class SpeedEstimator(BaseSolution):
    """
    A class to estimate the speed of objects in a real-time video stream based on their tracks.

    This class extends the BaseSolution class and provides functionality for estimating object speeds using
    tracking data in video streams.

    Attributes:
        spd (Dict[int, float]): Dictionary storing speed data for tracked objects.
        trkd_ids (List[int]): List of tracked object IDs that have already been speed-estimated.
        trk_pt (Dict[int, float]): Dictionary storing previous timestamps for tracked objects.
        trk_pp (Dict[int, Tuple[float, float]]): Dictionary storing previous positions for tracked objects.
        annotator (Annotator): Annotator object for drawing on images.
        region (List[Tuple[int, int]]): List of points defining the speed estimation region.
        track_line (List[Tuple[float, float]]): List of points representing the object's track.
        r_s (LineString): LineString object representing the speed estimation region.
        speeds (List[Tuple[str, float]]): List to store the estimated speeds along with vehicle types.

    Methods:
        initialize_region: Initializes the speed estimation region.
        estimate_speed: Estimates the speed of objects based on tracking data.
        store_tracking_history: Stores the tracking history for an object.
        extract_tracks: Extracts tracks from the current frame.
        display_output: Displays the output with annotations.
        save_speeds_to_file: Saves the estimated speeds to a text file and calculates the average speed for each vehicle type.

    Examples:
        >>> estimator = SpeedEstimator()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = estimator.estimate_speed(frame)
        >>> cv2.imshow("Speed Estimation", processed_frame)
    """

    def __init__(self, **kwargs):
        """Initializes the SpeedEstimator object with speed estimation parameters and data structures."""
        super().__init__(**kwargs)
        self.initialize_region()  # Initialize speed region
        self.spd = {}  # set for speed data
        self.trkd_ids = []  # list for already speed_estimated and tracked ID's
        self.trk_pt = {}  # set for tracks previous time
        self.trk_pp = {}  # set for tracks previous point
        self.speeds = []  # List to store speeds along with vehicle types

    def estimate_speed(self, im0):
        """
        Estimates the speed of objects based on tracking data.

        Args:
            im0 (np.ndarray): Input image for processing. Shape is typically (H, W, C) for RGB images.

        Returns:
            (np.ndarray): Processed image with speed estimations and annotations.

        Examples:
            >>> estimator = SpeedEstimator()
            >>> image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> processed_image = estimator.estimate_speed(image)
        """
        self.annotator = Annotator(
            im0, line_width=self.line_width
        )  # Initialize annotator
        self.extract_tracks(im0)  # Extract tracks

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 1
        )  # Draw region

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)  # Store track history

            # Check if track_id is already in self.trk_pp or trk_pt initialize if not
            if track_id not in self.trk_pt:
                self.trk_pt[track_id] = 0
            if track_id not in self.trk_pp:
                self.trk_pp[track_id] = self.track_line[-1]

            speed_label = (
                f"{int(self.spd[track_id])} km/h"
                if track_id in self.spd
                else self.names[int(cls)]
            )
            self.annotator.box_label(
                box, label=speed_label, color=colors(track_id, True)
            )  # Draw bounding box

            # Draw tracks of objects
            self.annotator.draw_centroid_and_tracks(
                self.track_line,
                color=colors(int(track_id), True),
                track_thickness=self.line_width,
            )

            # Calculate object speed and direction based on region intersection
            if self.LineString([self.trk_pp[track_id], self.track_line[-1]]).intersects(
                self.r_s
            ):
                direction = "known"
            else:
                direction = "unknown"

            # Perform speed calculation and tracking updates if direction is valid
            if direction == "known" and track_id not in self.trkd_ids:
                self.trkd_ids.append(track_id)
                time_difference = time() - self.trk_pt[track_id]
                if time_difference > 0:
                    speed = (
                        np.abs(self.track_line[-1][1] - self.trk_pp[track_id][1])
                        / time_difference
                    )
                    self.spd[track_id] = speed
                    vehicle_type = self.names[int(cls)]
                    self.speeds.append(
                        (vehicle_type, speed)
                    )  # Store the vehicle type and speed

            self.trk_pt[track_id] = time()
            self.trk_pp[track_id] = self.track_line[-1]

        self.display_output(im0)  # display output with base class function

        return im0  # return output image for more usage

    # def save_speeds_to_file(self, filename):
    #     """
    #     Saves the estimated speeds to a text file and calculates the average speed for each vehicle type.

    #     Args:
    #         filename (str): The name of the file to save the speeds to.
    #     """
    #     vehicle_speeds = defaultdict(list)

    #     # Write the speeds to the file and collect speeds for each vehicle type
    #     with open(filename, "w") as f:
    #         f.write("Vehicle Type      Speed (km/h)\n")
    #         f.write("==============================\n")
    #         for vehicle_type, speed in self.speeds:
    #             f.write(f"{vehicle_type:<15} {speed:>10.2f} km/h\n")
    #             vehicle_speeds[vehicle_type].append(speed)

    #     # Calculate and write the average speed for each vehicle type
    #     with open(filename, "a") as f:
    #         f.write("\nAverage Speeds:\n")
    #         f.write("==============================\n")
    #         for vehicle_type, speeds in vehicle_speeds.items():
    #             average_speed = sum(speeds) / len(speeds)
    #             f.write(f"{vehicle_type:<15} {average_speed:>10.2f} km/h\n")

    def save_speeds_to_file(self):
        """
        Saves the estimated speeds to a text file and calculates the average speed for each vehicle type.
        """
        vehicle_speeds = defaultdict(list)
        unique_id = uuid.uuid4()
        filename = f"output/speeds_{unique_id}.txt"

        # Write the speeds to the file and collect speeds for each vehicle type
        with open(filename, "w", encoding="utf-8") as f:
            f.write("Vehicle Type      Speed (km/h)\n")
            f.write("==============================\n")
            for vehicle_type, speed in self.speeds:
                if speed > 100:
                    f.write(
                        f"{vehicle_type:<13} {speed:>10.2f} km/h ⭕\n"
                    )  # Red marker for speeds > 100
                else:
                    f.write(f"{vehicle_type:<15} {speed:>10.2f} km/h\n")
                vehicle_speeds[vehicle_type].append(speed)

        # Calculate and write the average speed for each vehicle type
        with open(filename, "a", encoding="utf-8") as f:
            f.write("\nAverage Speeds:\n")
            f.write("==============================\n")
            for vehicle_type, speeds in vehicle_speeds.items():
                average_speed = sum(speeds) / len(speeds)
                f.write(f"{vehicle_type:<15} {average_speed:>10.2f} km/h\n")
