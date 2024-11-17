import cv2
import tkinter as tk
from tkinter import messagebox

# Main methoid to capture image samples
def captureImgSample(img, current_img_id):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = faceCascade.detectMultiScale(gray_img, 1.1, 10)
    coordinate = []
    check = 0

    # Draw rectangle around the faces
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, "Face", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
        coordinate = [x, y, w, h]

    # Save the image
    if len(coordinate) == 4:
        roi_img = img[coordinate[1]:coordinate[1] + coordinate[3], coordinate[0]:coordinate[0] + coordinate[2]]
        user_id = 1
        check = 1
        try:
            cv2.imwrite(f"Images/Jerard/Image.{user_id}.{current_img_id}.jpg", roi_img)
        except Exception as e:
            print(f"Error saving image: {e}")
            check = 0
    return check

# Load the cascade classifier
try:
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if faceCascade.empty():
        raise Exception("Error loading cascade classifier")
except Exception as e:
    print(f"Error loading cascade classifier: {e}")
    exit(1)

# Load the classifier model
try:
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.yml")
except Exception as e:
    print(f"Error loading classifier: {e}")
    exit(1)

# For the GUI
root = tk.Tk()
root.title("Capture Image Sample")
root.geometry("400x200")

status_label = tk.Label(root, text="Press Start to begin capturing", font=('Arial', 12))
status_label.pack(pady=20)

# Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    messagebox.showerror("Error", "Could not open camera")
    root.destroy()
    exit(1)

global img_id, is_capturing
img_id = 1
is_capturing = False

# Update the camera feed
def update_camera():
    global img_id, is_capturing
    if is_capturing and img_id < 51:
        success, img = cap.read()
        if success:
            check = captureImgSample(img, img_id)
            cv2.imshow("Testing", img)
            if check == 1:
                img_id += 1
                status_label.config(text=f"Captured {img_id-1} images")
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cleanup()
        root.after(10, update_camera)
    elif img_id >= 51:
        status_label.config(text="Capture Complete!")
        is_capturing = False
        start_button.config(state='disabled')
    elif is_capturing:
        root.after(10, update_camera)

# Start capturing images
def start_capture():
    global is_capturing
    is_capturing = True
    start_button.config(state='disabled')
    status_label.config(text="Capturing images...")
    update_camera()

# Cleanup and exit
def cleanup():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

# GUI buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=20)

# This is the start button to capture 50 images
start_button = tk.Button(button_frame, text="Start", command=start_capture, 
                        width=10, height=2, font=('Arial', 10), bg='green', disabledforeground='grey')
start_button.pack(side=tk.LEFT, padx=10)

# This is the exit button to close the program
exit_button = tk.Button(button_frame, text="Exit", command=cleanup, 
                       width=10, height=2, font=('Arial', 10), bg='red')
exit_button.pack(side=tk.LEFT, padx=10)

root.mainloop()
