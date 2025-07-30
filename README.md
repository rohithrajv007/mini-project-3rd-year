# 🎙️ Hate Speech Detection and Censorship System

The **Hate Speech Detection and Censorship System** is a privacy-first, real-time audio processing tool that leverages advanced speech recognition and censorship mechanisms to detect and mute hate speech in user-uploaded audio files. Unlike cloud-based alternatives, this system operates entirely offline, ensuring security and performance.

---

## 📌 Abstract

This system provides a fast and secure alternative to typical cloud-based audio censorship services. By using **OpenAI Whisper** for transcription and **Vosk** for timestamping, the project achieves ~95% transcription accuracy and 10x faster processing speed. All operations are performed locally, maintaining user privacy while delivering scalable and accurate results.

---

## ⚙️ System Architecture

- **Web Framework:** Flask-based web UI for audio upload, playback, and result viewing
- **Speech Recognition:** Whisper (for transcription), Vosk (for word-level timestamping)
- **Censorship Options:** Apply **beep** or **silence** using pydub
- **Detection:** Matches against a predefined list of offensive terms
- **Timestamp Refinement:** dB-based analysis using NumPy
- **Performance:** Asynchronous processing, LRU caching
- **Execution:** 100% local (no external API calls)

---

## 💻 Technology Stack

| Layer               | Technology Used                         |
|---------------------|------------------------------------------|
| Speech Recognition  | OpenAI Whisper, Vosk                     |
| Audio Processing    | pydub, NumPy                             |
| Web Application     | Flask                                    |
| Concurrency         | `concurrent.futures`, `functools.lru_cache` |
| Deployment          | Python 3.8+, Windows/macOS/Linux         |

---

## 🛠️ Features

- ✅ Audio upload via web UI  
- ✅ Choose censorship style: Beep 🔊 or Silence 🤫  
- ✅ Word-level timestamping  
- ✅ Downloadable transcript & censored audio  
- ✅ 100% offline privacy-first processing  
- ✅ Scalable design with plans for multilingual support  

---

## 📂 Directory Structure

hate-speech-detection/
├── app/
│ ├── templates/
│ ├── static/
│ ├── utils/
├── models/
├── uploads/
├── outputs/
├── requirements.txt
├── README.md
└── run.py

yaml
Copy
Edit

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/hate-speech-detection.git
cd hate-speech-detection
2. Set Up Python Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
pip install -r requirements.txt
3. Download Pretrained Models
Whisper: https://github.com/openai/whisper

Vosk: https://alphacephei.com/vosk/models

4. Run the Application
bash
Copy
Edit
python run.py
Open your browser and go to http://127.0.0.1:5000

📈 Future Improvements
🌐 Multilingual support

🧠 NLP-based contextual filtering

📊 Analytics dashboard

🤝 Acknowledgements
OpenAI Whisper

Vosk Speech Recognition

pydub

Flask

NumPy

