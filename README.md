# Corporate App

A Flask-based web application with integrated support for object detection (using yolo) and other capabilities like database management and video processing.

---

## 📑 Table of Contents

- [🚀 Features](#-features)
- [✅ Prerequisites](#-prerequisites)
- [💻 Installation](#-installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create a Virtual Environment](#2-create-a-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [⚡ Handling CUDA Version Differences](#-handling-cuda-version-differences)
- [🗄️ Database Setup](#️-database-setup)
  - [1. Configure Database](#1-configure-database)
  - [2. Initialize and Migrate Database](#2-initialize-and-migrate-database)
  - [3. Upload Existing Database](#3-upload-existing-database)
- [🏃‍♂️ Running the Project](#️-running-the-project)
- [🛠️ Configuring Apache](#️-configuring-apache)
- [📂 Project Structure](#-project-structure)
- [📜 License](#-license)
- [✨ Credits](#-credits)

---

## 🚀 Features

- Flask-based web server
- SQLAlchemy ORM with Flask-Migrate
- User authentication via Flask-Login
- Face detection and recognition:
  - MTCNN for detection
  - facenet-pytorch for embeddings
  - yolo for object detection
- OpenCV integration-ready
- Optional GPU acceleration (PyTorch + CUDA)
- Exportable reports (e.g., XLSX)

---

## 💻 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

Replace the URL with your actual repository.

---

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate:

- **Windows**
  ```bash
  .\venv\Scripts\activate
  ```
- **macOS/Linux**
  ```bash
  source venv/bin/activate
  ```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Includes:

- Flask
- Flask-Login, Flask-Migrate, Flask-SQLAlchemy
- facenet-pytorch
- torch, torchvision, torchaudio (CUDA 12.6 default)
- OpenCV
- Pandas, SQLAlchemy, etc.
- ultralystics

---

### ⚡ Handling CUDA Version Differences

By default:

```
torch==2.7.1+cu126
torchvision==0.22.1+cu126
torchaudio==2.7.1+cu126
```

✅ For **CPU-only**, replace in `requirements.txt`:

```
torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

✅ For other CUDA versions:

```
torch==2.7.1+cu118
torchvision==0.22.1+cu118
torchaudio==2.7.1+cu118
```

Use [PyTorch Get Started](https://pytorch.org/get-started/locally/) for exact commands.

---

## 🗄️ Database Setup

Uses **Flask-SQLAlchemy** and **Flask-Migrate**.

---

### 1. Configure Database

Edit `config.py`:

```python
import os
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'bismillah-ta'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'mysql+pymysql://root:@localhost/cctv_db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = os.path.join(basedir, 'app', 'static', 'img')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    HAARCASCADES_PATH = os.path.join(basedir, 'app', 'static', 'haarcascades')
    TRAINED_MODELS_PATH = os.path.join(basedir, 'app', 'static', 'trained_models')
    YOLO_MODELS_PATH = os.path.join(basedir, 'app', 'static', 'yolo')

    REMEMBER_COOKIE_DURATION = timedelta(days=7)
    PERMANENT_SESSION_LIFETIME = timedelta(days=30)
```

✅ Alternative Examples:

- **MySQL**
  ```
  SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://user:password@host:port/dbname'
  ```
- **PostgreSQL**
  ```
  SQLALCHEMY_DATABASE_URI = 'postgresql://user:password@host:port/dbname'
  ```

---

### 2. Initialize and Migrate Database

```bash
flask db init       # Once only
flask db migrate -m "Initial migration"
flask db upgrade
```

For schema changes:

```bash
flask db migrate -m "Describe change"
flask db upgrade
```

---

### 3. Upload Existing Database

✅ **MySQL**:

```bash
mysql -u username -p database_name < /path/to/database.sql
```

✅ **PostgreSQL**:

```bash
psql -U username -d database_name -f /path/to/database.sql
```

✅ **SQLite**:

```bash
cp /path/to/database.db /path/to/your/project/app.db
```

---

## 🏃‍♂️ Running the Project

Activate environment:

- **Windows**
  ```bash
  .\venv\Scripts\activate
  ```
- **macOS/Linux**
  ```bash
  source venv/bin/activate
  ```

Set app:

- **Windows**
  ```bash
  set FLASK_APP=run.py
  ```
- **macOS/Linux**
  ```bash
  export FLASK_APP=run.py
  ```

Run:

```bash
flask run
```

Open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🛠️ Configuring Apache

✅ Install mod_wsgi:

```bash
sudo apt-get install libapache2-mod-wsgi-py3
```

✅ Create `myapp.wsgi`:

```python
import sys
import os

sys.path.insert(0, '/path/to/your/project')
from run import app as application
```

✅ Example Apache Config:

```apache
<VirtualHost *:80>
    ServerName yourdomain.com
    WSGIDaemonProcess myapp user=yourusername group=yourgroup threads=5
    WSGIScriptAlias / /path/to/your/project/myapp.wsgi

    <Directory /path/to/your/project>
        Require all granted
    </Directory>

    ErrorLog ${APACHE_LOG_DIR}/myapp_error.log
    CustomLog ${APACHE_LOG_DIR}/myapp_access.log combined
</VirtualHost>
```

✅ Enable & restart:

```bash
sudo a2ensite myapp
sudo systemctl restart apache2
```

---

## 📂 Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── models.py
│   ├── routes.py
│   ├── templates/
│   └── static/
├── migrations/
├── config.py
├── requirements.txt
├── run.py
└── README.md
```

---

## 📜 License

This project is licensed under the MIT License. See LICENSE for details.

---

## ✨ Credits

- facenet-pytorch
- Flask
- PyTorch
- OpenCV
- ultralystics yolo


