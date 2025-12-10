import os
import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import sys

# -----------------------------
# DATASET OUTPUT DIRECTORY
# -----------------------------
OUTPUT_DIR = "dataset"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
LBL_DIR = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

last_valid_face_coord = ()

def get_frame_at(index, video):
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    last_frame_index = total_frames - 2

    if index > last_frame_index:
        index = last_frame_index

    video.set(cv2.CAP_PROP_POS_FRAMES, index)
    _, frame = video.read()
    return frame

def compute_coordinates(frame, face_detection, face_mesh):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w = frame.shape[:2]

    results = face_detection.process(image)

    global last_valid_face_coord
    curr_coord = last_valid_face_coord

    if results.detections:
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box

        x = bboxC.xmin * img_w
        y = bboxC.ymin * img_h
        w = bboxC.width * img_w
        h = bboxC.height * img_h

        # keep tracking_x, tracking_y from previous coord
        curr_coord = (x, y, w, h, last_valid_face_coord[4], last_valid_face_coord[5])

    last_valid_face_coord = curr_coord
    return curr_coord

def save_yolo_label(x, y, w, h, img_w, img_h, output_txt_path):
    """Converts pixel coords to YOLO normalized format and saves."""
    # clamp
    x = max(0, min(x, img_w))
    y = max(0, min(y, img_h))
    w = max(0, min(w, img_w))
    h = max(0, min(h, img_h))

    # convert to center format
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    label = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

    with open(output_txt_path, "w") as f:
        f.write(label)


def save_frame_as_dataset(frame, coords, video_name, frame_index):
    """Save frame + YOLO label."""
    img_h, img_w = frame.shape[:2]

    x, y, w, h, _, _ = coords

    # Create deterministic file names
    base_name = f"{video_name}_{frame_index}"

    img_path = os.path.join(IMG_DIR, base_name + ".jpg")
    lbl_path = os.path.join(LBL_DIR, base_name + ".txt")

    # Save frame
    cv2.imwrite(img_path, frame)

    # Save YOLO label
    save_yolo_label(x, y, w, h, img_w, img_h, lbl_path)


def smoothening_transition_interval(start_index, end_index, video, face_detection, face_mesh, filename):
    start_frame = get_frame_at(start_index, video)
    start_coords = compute_coordinates(start_frame, face_detection, face_mesh)

    end_frame = get_frame_at(end_index, video)
    end_coords = compute_coordinates(end_frame, face_detection, face_mesh)

    total_frames = end_index - start_index

    for curr_index in range(start_index, end_index):
        if total_frames == 0:
            continue

        # interpolation
        t = (curr_index - start_index) / total_frames
        curr_x = start_coords[0] + t * (end_coords[0] - start_coords[0])
        curr_y = start_coords[1] + t * (end_coords[1] - start_coords[1])
        curr_w = start_coords[2] + t * (end_coords[2] - start_coords[2])
        curr_h = start_coords[3] + t * (end_coords[3] - start_coords[3])

        frame = get_frame_at(curr_index, video)

        # save as YOLO dataset
        save_frame_as_dataset(frame, (curr_x, curr_y, curr_w, curr_h, 0, 0), filename[:-4], curr_index)


def select_input_folder():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select Input Folder")
    return folder_selected

def process_video_with_progress(video_path, filename, root, progress_label, progress_bar, static_label):
    video = cv2.VideoCapture(video_path)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    global last_valid_face_coord
    last_valid_face_coord = (0, 0, width, height, width // 2, height // 2)

    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        min_detection_confidence=0.3, model_selection=0
    )

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        min_detection_confidence=0.3, min_tracking_confidence=0.3
    )

    smoothening_interval = 1

    static_label.config(text=f"Processing: {filename}")
    progress_bar['value'] = 0
    progress_bar['maximum'] = total_frames

    def update_progress(current_frame):
        progress_bar['value'] = current_frame
        progress_label.config(text=f"{(current_frame / total_frames) * 100:.0f}% Complete")
        root.update_idletasks()

    for start_frame_index in range(0, total_frames, smoothening_interval):
        end_index = start_frame_index + smoothening_interval

        update_progress(start_frame_index)
        smoothening_transition_interval(start_frame_index, end_index, video, face_detection, face_mesh, filename)

    video.release()
    root.update_idletasks()

def create_progress_window():
    root = tk.Tk()
    root.title("Video Processing")
    root.geometry("500x200")
    root.configure(bg="#f9f9f9")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLabel", background="#f9f9f9", font=("Arial", 12), padding=5)
    style.configure("Header.TLabel", font=("Arial", 14, "bold"), foreground="#333333")
    style.configure("TProgressbar", thickness=10, troughcolor="#e0e0e0", background="#4caf50")

    static_label = ttk.Label(root, text="Initializing...", style="Header.TLabel")
    static_label.pack(pady=10)

    progress_label = ttk.Label(root, text="0%", style="TLabel")
    progress_label.pack(pady=10)

    progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate", style="TProgressbar")
    progress_bar.pack(pady=20)

    return root, static_label, progress_label, progress_bar

def main():
    input_dir = select_input_folder()
    if not input_dir:
        print("No folder selected. Exiting...")
        return

    root, static_label, progress_label, progress_bar = create_progress_window()

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            input_path = os.path.join(input_dir, filename)
            process_video_with_progress(input_path, filename, root, progress_label, progress_bar, static_label)

    static_label.config(text="All videos processed!")
    progress_label.config(text="100% Complete")
    root.after(2000, root.destroy)

    root.mainloop()

def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

if __name__ == "__main__":
    main()
