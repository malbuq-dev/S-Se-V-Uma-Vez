import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

from model import Yolov1
from utils import non_max_suppression, cellboxes_to_boxes


# ---------------------------
# Settings
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_WEIGHTS = "lia_v1.pth.tar"

# 0 = default webcam; or set to "path/to/video.mp4"
VIDEO_SOURCE = "test.mp4"

IMG_SIZE = 448
SPLIT_SIZE = 7
NUM_BOXES = 2
NUM_CLASSES = 1

VOC_CLASSES = [
    "person"
]

# Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Random colors for classes
COLORS = np.random.uniform(0, 255, size=(NUM_CLASSES, 3))


def load_model():
    """
    Load YOLOv1 model and weights into memory.
    """
    print(f"Loading model from {MODEL_WEIGHTS} on {DEVICE}...")
    ckpt = torch.load(MODEL_WEIGHTS, map_location=DEVICE)

    # Be robust whether file has {"state_dict": ...} or just the dict
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    model = Yolov1(
        split_size=SPLIT_SIZE,
        num_boxes=NUM_BOXES,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Model loaded.")
    return model


def draw_boxes(original_img, boxes, resized_w=IMG_SIZE, resized_h=IMG_SIZE):
    """
    Draw YOLOv1 boxes (normalized midpoint x,y,w,h) on the ORIGINAL image.
    """
    h, w = original_img.shape[:2]
    scale_x = w / float(resized_w)
    scale_y = h / float(resized_h)

    for box in boxes:
        class_pred = int(box[0])
        score = float(box[1])

        x, y, bw, bh = box[2], box[3], box[4], box[5]

        # Convert to pixel positions on resized image
        xmin_r = (x - bw / 2.0) * resized_w
        ymin_r = (y - bh / 2.0) * resized_h
        xmax_r = (x + bw / 2.0) * resized_w
        ymax_r = (y + bh / 2.0) * resized_h

        # Scale to original resolution
        xmin = int(xmin_r * scale_x)
        ymin = int(ymin_r * scale_y)
        xmax = int(xmax_r * scale_x)
        ymax = int(ymax_r * scale_y)

        if class_pred < 0 or class_pred >= NUM_CLASSES:
            continue  # safety check

        color = COLORS[class_pred]

        cv2.rectangle(original_img, (xmin, ymin), (xmax, ymax), color, 2)
        label = f"{VOC_CLASSES[class_pred]} {score:.2f}"
        cv2.putText(
            original_img,
            label,
            (xmin, max(0, ymin - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return original_img


def run_live():
    model = load_model()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: could not open video source {VIDEO_SOURCE}")
        return

    print("Starting live detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or cannot grab frame.")
            break

        # Convert BGR (OpenCV) to RGB and preprocess
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            predictions = model(img_tensor)

            all_bboxes = cellboxes_to_boxes(predictions)[0]   # list of [cls, score, x, y, w, h]

            bboxes = non_max_suppression(
                all_bboxes,
                iou_threshold=0.5,
                threshold=0.1,          # lower this while debugging
                box_format="midpoint"   # your boxes are [x_mid, y_mid, w, h]
            )

            print("Boxes after NMS:", len(bboxes))
            print("Sample NMS boxes (first 5):", bboxes[:5])

        # Draw final boxes on ORIGINAL frame
        frame_with_boxes = draw_boxes(frame.copy(), bboxes)


        cv2.imshow("YOLOv1 Live Detection", frame_with_boxes)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    run_live()
