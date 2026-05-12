# 🧠 Microwave Medical Imaging — Brain Stroke Diagnosis

A full-stack AI-powered diagnostic web application that detects and classifies brain strokes from microwave brain scan images using Machine Learning and image segmentation.

---

## 📌 Overview

This project addresses the critical challenge of early brain stroke detection by combining **microwave imaging technology** with **machine learning classification** and **DBIM-based image segmentation**. The system provides clinicians with a fast, intuitive web interface to upload scan images and receive instant diagnostic results.

---

## 🚀 Features

- 🔍 **Stroke Detection** — Random Forest classifier trained on 4,000+ labelled microwave brain scan images
- 🧩 **Advanced Segmentation** — DBIM Segmentation replaces traditional OTSU Thresholding for significantly higher accuracy
- 🌐 **Full-Stack Web App** — Django backend with a responsive, clinician-friendly frontend
- ⚡ **End-to-End Pipeline** — Image upload → preprocessing → ML inference → result rendering, all in one flow
- 📊 **Real-Time Results** — Instant diagnostic output displayed on the interface after upload

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| ML Model | Random Forest Classifier |
| Image Processing | OpenCV, DBIM Segmentation |
| Backend | Django |
| Frontend | HTML5, CSS3, JavaScript |
| Dataset | 4,000+ microwave brain scan images |

---

## 📁 Project Structure

```
brain-stroke-diagnosis/
├── manage.py
├── requirements.txt
├── README.md
├── app/
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   └── ml/
│       ├── model.pkl          # Trained Random Forest model
│       ├── segmentation.py    # DBIM Segmentation logic
│       └── preprocess.py      # Image preprocessing pipeline
├── templates/
│   ├── index.html
│   └── result.html
└── static/
    ├── css/
    └── js/
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/harshapemmadi00/brain-stroke-diagnosis.git
cd brain-stroke-diagnosis

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run database migrations
python manage.py migrate

# 5. Start the development server
python manage.py runserver
```

Then open your browser and go to: `http://127.0.0.1:8000`

---

## 🧪 How It Works

1. **Upload** a microwave brain scan image via the web interface
2. **Preprocessing** — OpenCV cleans and normalises the image
3. **Segmentation** — DBIM Segmentation isolates regions of interest
4. **Classification** — Random Forest model predicts stroke presence and type
5. **Result** — Diagnostic output is displayed instantly on screen

---

## 📊 Model Details

| Property | Detail |
|---|---|
| Algorithm | Random Forest Classifier |
| Training Data | 4,000+ labelled microwave brain scan images |
| Segmentation | DBIM (replacing OTSU Thresholding) |
| Task | Binary + multi-class stroke classification |

---

## 📸 Screenshots

> *(Add screenshots of your web interface here)*
> `![Upload Page](static/screenshots/upload.png)`
> `![Result Page](static/screenshots/result.png)`

---

## 🔮 Future Improvements

- [ ] Add support for DICOM image format
- [ ] Integrate deep learning model (CNN) for improved accuracy
- [ ] Deploy on cloud (AWS / Render)
- [ ] Add patient record management

---

## 👤 Author

**Pemmadi Harsha Vardhan Kumar**
- 📧 harshavardhan.p.236@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/harshapemmadi00)
- 🐙 [GitHub](https://github.com/harshapemmadi00)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
