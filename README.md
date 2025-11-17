# 🛡️ Real-Time Privacy Blur (Face Anonymization)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8--Face-green)
![OpenCV](https://img.shields.io/badge/Library-OpenCV-red)

A high-performance computer vision tool that automatically detects and blurs faces in real-time video streams or CCTV footage. Designed to ensure compliance with privacy regulations (GDPR/PDP) by anonymizing individuals instantaneously.

## 🎥 Live Demo
| Original | Cencored |
|:---:|:---:|
|![uncencored-gif](https://github.com/user-attachments/assets/075a8def-9872-460d-abf2-bda4651d00eb)|![cencored-gif](https://github.com/user-attachments/assets/bec7d945-d9e9-4cee-899c-2ae01a4dfd31)|


## 🌟 Key Features
* **High Accuracy:** Uses **YOLOv8-Face** model (better than Haar Cascades) to detect faces in various angles and lighting conditions.
* **Dynamic Blurring:** The blur intensity automatically adjusts based on the distance of the face from the camera.
* **Low Latency:** Optimized for real-time usage (Webcam/CCTV).

## 🛠️ Installation

1.  **Clone the repo**
    ```bash
    git clone [https://github.com/USERNAME_TUAN/realtime-privacy-blur.git](https://github.com/USERNAME_TUAN/realtime-privacy-blur.git)
    cd realtime-privacy-blur
    ```

2.  **Install requirements**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the tool**
    main.py to use video you want OR
    ```bash
    python main.py

    ```
    app.py to use webcam
    ```bash
    python app.py

    ```

## 🧠 How It Works
The script captures video frames, runs inference using a specialized YOLOv8 model fine-tuned for face detection, extracts the Region of Interest (ROI), applies a Gaussian Blur filter, and merges it back into the original frame. all within milliseconds.

---
