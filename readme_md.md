# Face Verification System with FaceNet

Sistem verifikasi wajah menggunakan FaceNet yang dapat dijalankan secara lokal dengan dukungan API dan Ngrok untuk akses eksternal.

## Fitur

- ✅ **FaceNet Model**: Menggunakan model FaceNet pre-trained untuk ekstraksi fitur wajah
- ✅ **MTCNN Face Detection**: Deteksi wajah yang akurat dan cepat
- ✅ **REST API**: API lengkap untuk verifikasi dan manajemen identitas
- ✅ **Ngrok Integration**: Akses eksternal melalui Ngrok tunnel
- ✅ **Dataset Processing**: Pemrosesan dataset otomatis dengan struktur folder
- ✅ **Real-time Verification**: Verifikasi wajah real-time melalui API
- ✅ **Testing Tools**: Tools untuk testing dan evaluasi sistem

## Struktur Project

```
face_verification/
├── config.py              # Konfigurasi sistem
├── face_detector.py       # MTCNN face detector
├── facenet_model.py       # FaceNet model wrapper
├── dataset_processor.py   # Pemrosesan dataset
├── train.py               # Script training
├── api_server.py          # Flask API server
├── test_verification.py   # Testing tools
├── requirements.txt       # Dependencies
├── README.md              # Dokumentasi
├── dataset/               # Folder dataset
│   ├── person1/
│   │   ├── person1_0001.jpg
│   │   ├── person1_0002.jpg
│   │   └── ...
│   ├── person2/
│   │   ├── person2_0001.jpg
│   │   └── ...
│   └── ...
├── models/                # Model dan embeddings
├── temp/                  # File temporary
└── logs/                  # Log files
```

## Instalasi

1. **Clone atau download project ini**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Buat struktur dataset:**
```
dataset/
├── John_Doe/
│   ├── john_doe_0001.jpg
│   ├── john_doe_0002.jpg
│   └── john_doe_0003.jpg
├── Jane_Smith/
│   ├── jane_smith_0001.jpg
│   ├── jane_smith_0002.jpg
│   └── jane_smith_0003.jpg
└── ...
```

**Format penting:**
- Satu folder per orang
- Nama folder = nama identitas
- Format file: `nama_orang_xxxx.jpg` (recommended)
- Support format: JPG, JPEG, PNG, BMP

## Penggunaan

### 1. Training Model

```bash
# Training dengan dataset default (./dataset)
python train.py

# Training dengan dataset custom
python train.py /path/to/your/dataset

# Training dengan threshold custom
python train.py --threshold 0.7
```

**Output training:**
- Model embeddings: `models/face_embeddings.pkl`
- Training log: `logs/training.log`

### 2. Menjalankan API Server

```bash
# Server lokal saja
python api_server.py

# Server dengan Ngrok (akses eksternal)
python api_server.py --ngrok

# Custom port
python api_server.py --port 8080

# Dengan Ngrok auth token
python api_server.py --ngrok --ngrok-token YOUR_TOKEN
```

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Status API dan info |
| GET | `/health` | Health check |
| POST | `/verify-face` | Verifikasi wajah |
| GET | `/identities` | List identitas terdaftar |
| POST | `/register-identity` | Registrasi identitas baru |
| DELETE | `/remove-identity` | Hapus identitas |

### 3. Testing Sistem

```bash
# Test single image
python test_verification.py path/to/image.jpg

# Test entire dataset
python test_verification.py --dataset ./dataset

# Interactive mode
python test_verification.py
```

## Contoh Penggunaan API

### Verifikasi Wajah

```bash
curl -X POST \
  http://localhost:5000/verify-face \
  -F "image=@/path/to/photo.jpg"
```

**Response sukses:**
```json
{
  "success": true,
  "message": "Face verification successful",
  "user_id": "John_Doe",
  "name": "John_Doe",
  "similarity": 0.856,
  "data": {
    "face_detected": true,
    "detection_confidence": 0.998,
    "processing_time_ms": 245.67,
    "distance": 0.342,
    "confidence": 85.6
  }
}
```

**Response gagal:**
```json
{
  "success": false,
  "message": "Face not recognized",
  "code": "FACE_NOT_RECOGNIZED",
  "best_match": "Jane_Smith",
  "data": {
    "face_detected": true,
    "detection_confidence": 0.995,
    "processing_time_ms": 198.34,
    "distance": 0.789,
    "confidence": 42.3
  }
}
```

### Registrasi Identitas Baru

```bash
curl -X POST \
  http://localhost:5000/register-identity \
  -F "name=New_Person" \
  -F "images=@photo1.jpg" \
  -F "images=@photo2.jpg" \
  -F "images=@photo3.jpg"
```

### List Identitas

```bash
curl http://localhost:5000/identities
```

### Hapus Identitas

```bash
curl -X DELETE \
  http://localhost:5000/remove-identity \
  -H "Content-Type: application/json" \
  -d '{"name": "Person_Name"}'
```

## Konfigurasi

Edit `config.py` untuk menyesuaikan pengaturan:

```python
class Config:
    # Image settings
    IMAGE_SIZE = (160, 160)  # FaceNet standard
    
    # Verification threshold
    VERIFICATION_THRESHOLD = 0.6  # Adjust based on your needs
    
    # API settings
    API_HOST = '0.0.0.0'
    API_PORT = 5000
    
    # Face detection
    MIN_FACE_SIZE = 40
    DETECTION_THRESHOLD = [0.6, 0.7, 0.7]  # MTCNN thresholds
```

## Performance Tips

1. **Threshold Tuning:**
   - Lower threshold (0.4-0.5): Lebih permisif, higher recall
   - Higher threshold (0.7-0.8): Lebih ketat, higher precision
   - Default 0.6: Balance yang baik

2. **Dataset Quality:**
   - Minimal 3-5 foto per orang
   - Variasi pose dan pencahayaan
   - Resolusi minimal 160x160 pixels
   - Wajah jelas dan tidak blur

3. **Hardware:**
   - GPU: Significantly faster (CUDA required)
   - CPU: Tetap bisa digunakan, lebih lambat
   - RAM: Minimal 4GB untuk model loading

## Troubleshooting

### Error "No module named 'torch'"
```bash
pip install torch torchvision
```

### Error "No face detected"
- Pastikan wajah terlihat jelas
- Cek pencahayaan foto
- Pastikan resolusi cukup tinggi

### Error "No trained model found"
```bash
python train.py
```

### Ngrok error
- Install pyngrok: `pip install pyngrok`
- Set auth token jika diperlukan

### Poor accuracy
- Tambah lebih banyak foto training
- Adjust threshold di config.py
- Pastikan kualitas foto baik

## Development

### Menambah Model Baru
Extend `FaceNetModel` class di `facenet_model.py`

### Custom Face Detector
Extend `FaceDetector` class di `face_detector.py`

### API Endpoints Baru
Tambahkan routes di `api_server.py`

## License

MIT License - Free for personal and commercial use.

## Support

Untuk pertanyaan atau issue, silakan buat GitHub issue atau hubungi developer.

---

**Happy Face Verification! 🚀👤🔍**