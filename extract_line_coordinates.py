import cv2
import sys

points = []
frame = None

def click_event(event, x, y, flags, params):
    global points, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        if len(points) == 2:
            cv2.line(frame, points[0], points[1], (0, 0, 255), 2)

        cv2.imshow("Select Line", frame)

        if len(points) == 2:
            print("\nYour center line coordinates:")
            print(points)
            print("\nPaste this into your code as:")
            print(f"self.center_line = {points}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_line_coordinates.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if not ret:
        print("Could not read video.")
        sys.exit(1)

    cv2.namedWindow("Select Line", cv2.WINDOW_NORMAL)
    cv2.imshow("Select Line", frame)
    cv2.setMouseCallback("Select Line", click_event)

    print("Click TWO points to define the line")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
