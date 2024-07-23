import numpy as np
import time
import cv2
import os

def anonymize_face_simple(image, factor=3.0):
    # Automatically determine the size of the blurring kernel based on the spatial dimensions of the input image
    (h, w) = image.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)

    # Ensure the width of the kernel is odd
    if kW % 2 == 0:
        kW -= 1

    # Ensure the height of the kernel is odd
    if kH % 2 == 0:
        kH -= 1

    # Apply a Gaussian blur to the input image using our computed kernel size
    return cv2.GaussianBlur(image, (kW, kH), 0)

def anonymize_face_pixelate(image, blocks=3):
    # Divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")

    # Loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # Compute the starting and ending (x, y)-coordinates for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]

            # Extract the ROI using NumPy array slicing, compute the mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (B, G, R), -1)

    # Return the pixelated blurred image
    return image

# Load serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join(['.', "deploy.prototxt"])
weightsPath = os.path.sep.join(['.', "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)
conf = 0.5
meth = 'pixelated'
blk = 10

# Saved output
output_folder_path = './processed'

# Create the output directory if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

folder_path = './video'
video_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
print('Video files: ', video_files)

for video_file in video_files:
    video_path = os.path.join(folder_path, video_file)
    cap = cv2.VideoCapture(video_path)
    time.sleep(2.0)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        continue

    # Get the width and height of the video frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(output_folder_path, video_file)
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"Processing video file: {video_file}")

    # Loop over the frames from the video
    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video reached ")
            break

        # Check if the frame is empty
        if frame is None or frame.size == 0:
            print("Error: Empty frame")
            break

        # Resize the frame (optional, if needed)
        frame = cv2.resize(frame, (width, height))

        # Grab the dimensions of the frame and then construct a blob from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Pass the blob through the network and obtain the face detections
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract the confidence (i.e., probability) associated with the detection
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
            if confidence > conf:
                # Compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Extract the face ROI
                face = frame[startY:endY, startX:endX]

                # Apply the selected face anonymization method
                if meth == "simple":
                    face = anonymize_face_simple(face, factor=3.0)
                else:
                    face = anonymize_face_pixelate(face, blocks=blk)

                # Store the blurred face in the output image
                frame[startY:endY, startX:endX] = face

        # Write the frame to the output video
        out.write(frame)

        # If the `q` key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Do a bit of cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
