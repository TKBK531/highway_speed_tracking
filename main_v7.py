from ultralytics import YOLO
import speed_estimation
import cv2

model = YOLO("yolo11n.pt")
names = model.model.names

cap = cv2.VideoCapture("highway.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

line_pts = [(0, 150), (1280, 350)]

speed_obj = speed_estimation.SpeedEstimator()
speed_obj.reg_pts = line_pts
speed_obj.names = names
speed_obj.view_img = True

while cap.isOpened():

    success, im0 = cap.read()
    if not success:
        print(
            "Video frame is empty or video processing has been successfully completed."
        )
        break

    tracks = model.track(im0, persist=True, show=False)

    im0 = speed_obj.estimate_speed(im0)  # Pass only the image

    # Draw the line on the frame
    # cv2.line(im0, line_pts[0], line_pts[1], (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Speed Estimation", im0)

    # Press 'q' to exit the video display window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Save the speeds to a file
speed_obj.save_speeds_to_file()
