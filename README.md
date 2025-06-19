# 🛰️ Klasifikasi Landscape Aerial dengan Deep Learning

> Sistem klasifikasi otomatis untuk mengenali 15 jenis landscape dari foto udara menggunakan CNN dan Transfer Learning

## 🌍 Overview Dataset

**Skyview Multi-Landscape Aerial Imagery Dataset** - Koleksi 12,000 gambar aerial berkualitas tinggi untuk riset computer vision.

### 📊 Spesifikasi Dataset

- **Total Kategori**: 15 jenis landscape
- **Gambar per Kategori**: 800 (seimbang sempurna)
- **Resolusi**: 256×256 pixels → diresize ke 128×128
- **Total Gambar**: 12,000
- **Sumber**: [Kaggle - Skyview Dataset](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)

### 🏞️ Kategori Landscape

| Agriculture |  Airport  |  Beach  |    City     |  Desert  |
| :---------: | :-------: | :-----: | :---------: | :------: |
|   Forest    | Grassland | Highway |    Lake     | Mountain |
|   Parking   |   Port    | Railway | Residential |  River   |

## 📊 Struktur Project

```
Deep-Learning-Image-Classification/
├── .github/
│   └── workflows/
│       └── classification_workflow.yml
├── Processed-Classification/
│   ├── Submission_Akhir.ipynb
│   ├── automate_image_classification.py
│   └── processed_data/
│       ├── finish_model_savedmodel/
│   	│	├── assets/
│   	│	├── variables/
│   	│	│	├── variables.data-00000-of-00001
│   	│	│	└── variables.index
│   	│	├── fingerprint.pb
│   	│	└── saved_model.pb
│       ├── finish_model_tfjs/
│   	│	├── group1-shard1of4.bin
│   	│	├── group1-shard2of4.bin
│   	│	├── group1-shard3of4.bin
│   	│	├── group1-shard4of4.bin
│   	│	└── model.json
│       ├── finish_model_tflite/
│   	│	├── finish_model.tflite
│   	│	└── labels.txt
│       └── finish_model.keras
├── README.md
└── requirements.txt
```

## 📝 Credits

- **Dataset**: [Skyview Aerial Dataset](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)
- **Original Sources**: AID Dataset & NWPU-Resisc45 Dataset
- **Architecture**: MobileNetV2 + Custom Layers
