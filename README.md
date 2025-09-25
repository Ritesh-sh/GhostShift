
# ghostshift-repo
# 👻 GhostShift (Thumbs Up Gesture)


This project uses **Python, OpenCV, and Mediapipe** to create a fun “GhostShift” effect.  
When you show a **thumbs up gesture** to the camera, you disappear smoothly with a fade effect and magical sound.  
Press **thumbs up again** to reappear.  

---

## ✨ Features

- 🖐️ Detects thumbs-up gesture using Mediapipe Hands  
- 🎭 Smooth invisibility effect with background blending  
- 🪄 Magical sound effect when toggling invisibility  
- ⏱️ Fade-in / fade-out animation for realism  
- ❌ Exit safely using `q` or `ESC`  

---

## 🛠️ Requirements

- Python **3.9+** (works best on 3.10/3.11)  

Install the dependencies:

```bash
pip install opencv-python mediapipe playsound==1.2.2 numpy


📂 Project Structure
ghostshift/
│
├── cloak.py              # Main Python script
├── magic_whoosh.wav      # Sound effect file (place here)
└── README.md             # Project documentation

How to Run (in VS Code)


Clone or copy this project into a folder (e.g., ghostshift).

Open the folder in Visual Studio Code.

Install dependencies in the terminal:

pip install opencv-python mediapipe playsound==1.2.2 numpy


Make sure the sound file is named magic_whoosh.wav and placed in the same folder as cloak.py.

Run the program in the VS Code terminal:

Controls

👍 Thumbs Up gesture → Toggle invisibility (disappear/reappear)

🎵 Magical sound plays on toggle (if magic_whoosh.wav is present)

❌ Press q or ESC to quit

created by chirag sharma





