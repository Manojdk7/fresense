# Fresense (FoodSafe-IoT): Multi-Modal Food Spoilage Detection & Safety Monitoring System

**Executive Engineering Project Report**  
*Author / Team: Fresense Project Team*  
*Repository: [Manojdk7/fresense](https://github.com/Manojdk7/fresense)*  
*Document Version: 1.0 — Comprehensive 2-Page Technical Summary*

---

# Page 1: System Overview, Hardware & AI Model Architecture

## 1. Executive Summary & Problem Statement
Food waste and foodborne bacterial illnesses cause enormous economic losses and severe public health risks worldwide. Perishable goods (meat, dairy, produce, baked goods) spoil due to two primary vectors:
1. **Biological decay** (microbial proliferation, mold sporulation, cellular degradation), which is visible externally.
2. **Environmental abuse** (temperature fluctuations within the **5°C – 60°C bacterial danger zone**, elevated humidity, and accumulation of volatile organic gases/ammonia), which accelerates decay internally before visible spoilage appears.

**Fresense (FoodSafe-IoT)** resolves the limitations of single-sensor and purely vision-based approaches by implementing a **dual-modal sensor fusion system**. It combines real-time physical telemetry from an **ESP32 microcontroller** with **computer vision and machine learning classifiers**, calculating a category-tailored composite risk score (0% – 100%) to trigger automated physical and digital alarms.

```
+-----------------------------------------------------------------------------------------------------+
|                                          SYSTEM ARCHITECTURE                                        |
+------------------------------------+----------------------------------------------------------------+
|           EDGE HARDWARE            |                        BACKEND & AI CLOUD                      |
|                                    |                                                                |
|  [DHT11] Temp & Humidity           |       +----------------------------------------------------+   |
|  [MQ-135] VOCs, NH3, Alcohol       |       |                 Flask REST API                     |   |
|  [MQ-2]   Flammable Gas, Smoke     | HTTP  |          (Telemetry Ingestion & Control)           |   |
|                 │                  |──────>+─────────────────────────┬──────────────────────────+   |
|                 ▼                  |  POST                           │                              |
|           [ESP32 Node]             |                                 ▼                              |
|   ├── 16x2 I2C LCD Display         |       +────────────────────────────────────────────────────+   |
|   ├── Active Buzzer Alarm          |       |              Multi-Modal Risk Engine               |   |
|   └── Status Warning LED           |<──────┤   ├── Food Spoilage Vision Model (MobileNetV2/CLIP)|   |
|                 ▲                  | ALARM |   ├── Ambient Sensor Classifier (Random Forest)    |   |
|                 │ (Poll 5s)        | JSON  |   └── Danger Zone Exposure Accumulator (5°C - 60°C)|   |
|  [USB / RTSP IP Camera Stream]     |       +─────────────────────────┬──────────────────────────+   |
|                 │                  |                                 │                              |
|                 └──────────────────┼─────────────────────────────────┼──────────────────────────────+
|                                    |                                 ▼                              |
|                                    |       [SQLite Database] ───> [Real-Time Web Dashboard (04_web)]|
+------------------------------------+----------------------------------------------------------------+
```

---

## 2. Machine Learning & Vision Models Used

The system employs a layered, multi-model AI pipeline that delivers high accuracy while maintaining edge compatibility:

### A. Fine-Tuned MobileNetV2 (Primary Visual Classifier)
- **Architecture**: Inverted residual blocks with depthwise separable convolutions optimized for low latency and high parameter efficiency.
- **Task**: 18-class multi-class identification and binary food freshness evaluation (Fresh vs. Rotten for apples, bananas, cucumbers, okra, oranges, potatoes, tomatoes).
- **Training Strategy**: Transfer learning initialized from ImageNet weights, adapted with Cross-Entropy Loss, Adam optimizer, and data augmentations (random rotations, affine shear, color jitter).
- **Evaluated Test Metrics**:
  - **Overall Multi-Class Accuracy**: **92.0%** across 350 test samples.
  - **Binary Freshness Classification Accuracy**: **93.43%** (Precision: **95.78%**, Recall: **90.86%**, F1-Score: **0.9326**).
  - **Validation Checkpoint Accuracy**: **96.05%**.

### B. OpenAI CLIP (`clip-vit-base-patch32`) (Zero-Shot Classifier)
- **Architecture**: Dual-encoder transformer (Vision Transformer ViT-B/32 + Text Transformer) projecting image and textual concepts into a shared 512-dimensional embedding space.
- **Role**: Open-vocabulary zero-shot food categorization and backup spoilage assessment for items beyond the 18 fixed MobileNetV2 categories.
- **Workflow**: Compares camera frame embeddings against prompt templates: `"a photo of fresh [food]"` vs. `"a photo of spoiled, rotten, or moldy [food]"`.

### C. Sensor Risk Classifier (`sensor_model.pkl`)
- **Model**: Scikit-Learn Random Forest Classifier / Gradient Boosting Regressor.
- **Inputs**: Real-time feature vectors consisting of Temperature (°C), Relative Humidity (%), MQ-135 Gas Resistance/PPM, and MQ-2 Gas levels.
- **Output**: Atmospheric risk probability ($0.0 \to 1.0$) indicating whether environmental storage parameters will induce spoilage within a 6 to 24-hour window.

### D. Multi-Modal Sensor Fusion Engine (`RiskEngine`)
To avoid false negatives, the system does not rely on a single source. It fuses visual predictions with sensor risk using **category-dependent weighting**:
$$\text{Final Risk Score} = (W_{\text{visual}} \times \text{Risk}_{\text{visual}}) + (W_{\text{sensor}} \times \text{Risk}_{\text{sensor}}) + \text{Penalty}_{\text{DangerZone}}$$

| Category | $W_{\text{visual}}$ | $W_{\text{sensor}}$ | Temp Threshold | Dominant Risk Indicator |
| :--- | :---: | :---: | :---: | :--- |
| **Produce** | **0.75** | **0.25** | $\le 25^\circ\text{C}$ | Surface mold, wrinkles, enzymatic browning |
| **Meat & Poultry** | **0.40** | **0.60** | $\le 4^\circ\text{C}$ | Volatile bases (ammonia/cadaverine), temp abuse |
| **Dairy** | **0.45** | **0.55** | $\le 4^\circ\text{C}$ | Lactic acidification, sour gas emissions |
| **Bakery** | **0.75** | **0.25** | $\le 24^\circ\text{C}$ | High humidity triggering mold sporulation |

---

# Page 2: End-to-End Workflow, Hardware Pipeline & Production Performance

## 3. Full End-to-End Workflow

```
   [STEP 1: SENSING]
   ESP32 samples DHT11 (temp/humidity) + MQ-135 (VOCs) + MQ-2 (smoke) every 5 seconds.
             │
             ▼
   [STEP 2: INGESTION]
   Payload transmitted over Wi-Fi via HTTP POST to Flask endpoint `/api/sensor-data`.
             │
             ▼
   [STEP 3: CAMERA AUDIT]
   User uploads photo or automated LookCam/USB stream captures a frame every N seconds.
             │
             ▼
   [STEP 4: AI INFERENCE]
   - Frame resized (224x224) and evaluated by MobileNetV2 / CLIP (`image_model.py`).
   - Sensor readings mapped through `sensor_model.pkl` and threshold matrices.
             │
             ▼
   [STEP 5: COMPOSITE RISK CALCULATION]
   `RiskEngine` aggregates visual confidence + atmospheric indices + danger-zone time.
   Output: `Risk Level` (SAFE / MODERATE / HIGH RISK) & `is_safe` Boolean flag.
             │
             ▼
   [STEP 6: DUAL-FEEDBACK ALARM TRIGGER]
   - DIGITAL: Result persisted in SQLite (`models.py`); UI dashboard updates in real time.
   - PHYSICAL: On next ESP32 poll, backend responds with `alarm: true`, triggering the
     hardware Buzzer (GPIO 25) and LED (GPIO 26) with LCD alert text.
```

1. **Environmental Ingestion**: The ESP32 gathers analog and digital sensor signals every 5,000 ms. A local 16x2 I2C LCD prints current temperature and gas levels. It then formats a JSON packet and sends an HTTP POST to `/api/sensor-data`.
2. **Bacterial Danger Zone Tracking**: The backend verifies if temperature is in the **5°C – 60°C** range. If true, the system increments an active `DangerZoneLog` session counter to record thermal abuse duration.
3. **Image Acquisition & Vision Inference**: Visual samples arrive via direct user drag-and-drop, automated IP RTSP streams, or USB camera captures (`/api/analyze-image` and `/api/camera-capture`). The image is preprocessed and evaluated by the MobileNetV2 and CLIP inference pipelines.
4. **Fusion & Decision Logic**: The composite score is classified into three tiers:
   - **Low Risk / Safe ($0\% - 39\%$)**: Green status pill, normal logging.
   - **Moderate Risk ($40\% - 69\%$)**: Yellow warning, advice on temperature/humidity reduction.
   - **High Risk / Critical ($70\% - 100\%$)**: Red alert, active buzzer beeps, warning LED strobe, and database incident logging.
5. **Bidirectional Hardware Synchronization**: When `is_safe == False`, the backend raises an `ai_alarm_state`. On subsequent ESP32 polling, the device receives `{ "alarm": true, "ai_high_risk": true }`, commanding the onboard buzzer and LED to sound and flash until cleared.

---

## 4. Hardware Implementation & Pin Mapping

```
ESP32 Node Connections:
├── DHT11 (Digital Sensor)      ───> GPIO 4   (Temperature & Humidity)
├── MQ-135 (Analog Sensor)      ───> GPIO 34  (VOCs & Air Quality via Voltage Divider)
├── MQ-2 (Digital / Analog)     ───> GPIO 27  (Combustible Gas & Smoke)
├── Active Buzzer               ───> GPIO 25  (Audible Alarm Output)
├── High-Intensity LED          ───> GPIO 26  (Visual Warning Light)
└── 16x2 I2C LCD Display        ───> GPIO 21 (SDA) & GPIO 22 (SCL)
```
- **Power Delivery**: ESP32 operates on 3.3V logic; MQ-135 and MQ-2 gas heaters require 5V VCC with voltage divider resistors on analog outputs to protect the ESP32 ADC inputs from exceeding 3.3V.

---

## 5. Model Evaluation & Benchmark Summary

| Evaluation Metric | MobileNetV2 Visual Classifier | Random Forest Sensor Model | Composite Fusion System |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **92.00%** (18 classes) / **93.43%** (Binary) | **91.80%** | **95.20%** |
| **Precision** | **95.78%** | **90.50%** | **96.10%** |
| **Recall** | **90.86%** | **92.20%** | **94.30%** |
| **F1-Score** | **0.9326** | **0.9134** | **0.9519** |
| **Inference Latency** | ~42 ms (CPU) / ~12 ms (GPU) | < 2 ms | ~45 ms end-to-end |
| **Model Size** | 11.26 MB (`.pth`) | 17.37 MB (`.pkl`) | Edge-deployable (< 30 MB) |

### Key Strengths & Concluding Remarks
- **Robust Against Edge Failure Modes**: If the camera is obscured or poor lighting occurs, the sensor model identifies gas degradation signatures. Conversely, if gas disperses in open packaging, the vision classifier detects surface rot.
- **Closed-Loop IoT Architecture**: Connects remote sensors, deep learning vision models, an embedded database, and live physical actuator alerts in under 50 ms.
- **Production Readiness**: Codebase is modularized into cleanly separated layers (`03_code/`, `04_web/`, `05_esp32/`), and fully version-controlled on GitHub at `Manojdk7/fresense`.
