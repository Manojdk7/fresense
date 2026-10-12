/*
 * ============================================================
 * FoodSafe-IoT — ESP32
 * DHT11 + MQ135 + MQ2 + Buzzer + Flask Backend
 * ============================================================
 *
 * HARDWARE:
 *
 * DHT11:
 *   VCC  -> 3.3V
 *   GND  -> GND
 *   DATA -> GPIO 4
 *
 * MQ135:
 *   VCC -> 5V/VIN
 *   GND -> GND
 *   AO  -> 10k resistor -> GPIO 34
 *   GPIO 34 -> 20k resistor -> GND
 *
 * MQ2:
 *   VCC -> 5V/VIN
 *   GND -> GND
 *   DO  -> 10k resistor -> GPIO 27
 *   GPIO 27 -> 20k resistor -> GND
 *
 * Buzzer:
 *   + -> GPIO 25
 *   - -> GND
 *
 * LED:
 *   + -> GPIO 26
 *   - -> GND (via resistor)
 *
 * POWER:
 *   ESP32 powered from laptop USB
 *
 * ============================================================
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ============================================================
// WIFI SETTINGS
// ============================================================

const char* WIFI_SSID = "manu";
const char* WIFI_PASSWORD = "manu@777";

// ============================================================
// FLASK BACKEND
// ============================================================

// MUST match the PC LAN IPv4 that runs Flask (run: ipconfig).
// This PC is currently 10.77.123.110.
// Re-upload this sketch after any Wi-Fi DHCP IP change.
const char* BACKEND_URL =
  "http://10.77.123.110:5000/api/sensors/data";

const char* DEVICE_ID = "ESP32_001";

// ============================================================
// PIN DEFINITIONS
// ============================================================

#define DHT_PIN       4
#define MQ135_PIN     34
#define MQ2_DO_PIN    27
#define BUZZER_PIN    25
#define LED_PIN       26

#define DHT_TYPE DHT11

// ============================================================
// MQ135 THRESHOLD
// ============================================================
//
// Normal room air baseline after warm-up is approximately 1600 - 2000.
// Threshold is set to 2500 so clean air does NOT falsely trigger the alarm,
// but smoke or heavy spoilage gas (> 2500) will sound the alarm.
//

#define MQ135_THRESHOLD 2500

// ============================================================
// MQ2 DIGITAL STATE
// ============================================================
//
// Most common MQ2 modules:
// DO = LOW  -> gas detected
// DO = HIGH -> normal
//
// If your module behaves opposite, change LOW to HIGH.
//

#define MQ2_GAS_STATE LOW

// ============================================================
// TIMING
// ============================================================

#define SEND_INTERVAL_MS 5000
#define BUZZER_DURATION_MS 7000
#define WIFI_RETRY_DELAY 1000
#define MAX_WIFI_RETRIES 20
#define HTTP_TIMEOUT_MS 10000

// ============================================================
// DHT OBJECT
// ============================================================

DHT dht(DHT_PIN, DHT_TYPE);
LiquidCrystal_I2C lcd(0x3F, 16, 2);

unsigned long lastSendTime = 0;

// Latest AI HIGH-RISK flag from Flask (updated on each successful POST response)
bool aiHighRisk = false;
int lastAiAlarmId = 0;

// Latest AI Scan Result for LCD
String lastAiScanClass = "";
float lastAiScanScore = -1.0;
bool lastAiScanIsSafe = true;
unsigned long scanResultShowUntil = 0; // Millis timestamp until which scan result is shown (e.g. 15s)

// Buzzer: one 7-second burst per alarm activation, then stays silent until alarm clears
unsigned long buzzerStopAt = 0;
bool buzzerSounding = false;
bool alarmWasActive = false;
unsigned long alarmClearSince = 0;

#define ALARM_CLEAR_DEBOUNCE_MS 500

// ============================================================
// LCD DISPLAY FUNCTIONS
// ============================================================

void updateLcdTelemetry(float temp, float hum, int gasIndex) {
  // Line 1: T: 28.5C  H: 62%
  lcd.setCursor(0, 0);
  lcd.print("T:");
  lcd.print(temp, 1);
  lcd.print("C H:");
  lcd.print((int)round(hum));
  lcd.print("%   ");

  // Line 2: Air: 94  SAFE
  lcd.setCursor(0, 1);
  lcd.print("Air:");
  lcd.print(gasIndex);
  if (gasIndex < 200) {
    lcd.print(" [SAFE]  ");
  } else {
    lcd.print(" [ALERT] ");
  }
}

void showLcdScanResult(String className, float score, bool isSafe) {
  lcd.clear();
  lcd.setCursor(0, 0);
  if (isSafe) {
    lcd.print(">> FRESH (SAFE)");
  } else {
    lcd.print(">> SPOILED!");
  }

  lcd.setCursor(0, 1);
  // Show score percentage: e.g. "Risk: 84% "
  int pct = (int)round(score * 100);
  lcd.print("Risk:");
  lcd.print(pct);
  lcd.print("% ");
  
  // Truncate class name to fit remaining LCD chars
  if (className.length() > 0) {
    String shortName = className;
    shortName.replace("FRESH_", "");
    shortName.replace("ROTTEN_", "");
    if (shortName.length() > 6) {
      shortName = shortName.substring(0, 6);
    }
    lcd.print(shortName);
  }
}

// ============================================================
// ALARM OUTPUT HELPERS
// ============================================================

void startBuzzer(unsigned long durationMs = BUZZER_DURATION_MS) {
  tone(BUZZER_PIN, 2000);
  buzzerStopAt = millis() + durationMs;
  buzzerSounding = true;
  Serial.print("BUZZER: ON for ");
  Serial.print(durationMs / 1000);
  Serial.println(" seconds");
}

void stopBuzzer() {
  noTone(BUZZER_PIN);
  digitalWrite(BUZZER_PIN, LOW);
  buzzerSounding = false;
  buzzerStopAt = 0;
}

void updateAlarmOutputs(
  bool mq135GasDetected,
  bool mq2GasDetected
) {
  bool finalAlarm =
    mq135GasDetected ||
    mq2GasDetected ||
    aiHighRisk;

  // Hard stop when buzzer window ends
  if (buzzerSounding &&
      buzzerStopAt != 0 &&
      (long)(millis() - buzzerStopAt) >= 0) {
    stopBuzzer();
    Serial.println("BUZZER: OFF (complete)");
  }

  if (finalAlarm) {
    alarmClearSince = 0;
    digitalWrite(LED_PIN, HIGH);

    // Rising edge for hardware sensors
    if (!alarmWasActive) {
      startBuzzer(BUZZER_DURATION_MS);
      alarmWasActive = true;
    }

  } else {
    // Require stable clear before allowing another sensor burst
    if (alarmClearSince == 0) {
      alarmClearSince = millis();
    }

    if (millis() - alarmClearSince >= ALARM_CLEAR_DEBOUNCE_MS) {
      if (!buzzerSounding) {
        stopBuzzer();
      }
      digitalWrite(LED_PIN, LOW);
      alarmWasActive = false;
    }
  }
}

// ============================================================
// SETUP
// ============================================================

void setup() {

  Serial.begin(115200);

  delay(1000);

  Serial.println();
  Serial.println("========================================");
  Serial.println("          FoodSafe-IoT");
  Serial.println("             ESP32");
  Serial.println("========================================");

  // ----------------------------------------------------------
  // LCD (I2C: SDA=21, SCL=22)
  // ----------------------------------------------------------

  Wire.begin(21, 22);
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("FoodSafe-IoT");
  lcd.setCursor(0, 1);
  lcd.print("Online...");

  // ----------------------------------------------------------
  // DHT11
  // ----------------------------------------------------------

  dht.begin();

  Serial.println("DHT11  : GPIO 4");

  // ----------------------------------------------------------
  // MQ135
  // ----------------------------------------------------------

  pinMode(MQ135_PIN, INPUT);

  Serial.println("MQ135  : GPIO 34");

  // ----------------------------------------------------------
  // MQ2
  // ----------------------------------------------------------

  pinMode(MQ2_DO_PIN, INPUT);

  Serial.println("MQ2 DO : GPIO 27");

  // ----------------------------------------------------------
  // BUZZER
  // ----------------------------------------------------------

  pinMode(BUZZER_PIN, OUTPUT);

  noTone(BUZZER_PIN);

  Serial.println("Buzzer : GPIO 25");

  // ----------------------------------------------------------
  // LED
  // ----------------------------------------------------------

  pinMode(LED_PIN, OUTPUT);

  digitalWrite(LED_PIN, LOW);

  Serial.println("LED    : GPIO 26");

  // ----------------------------------------------------------
  // ESP32 ADC
  // ----------------------------------------------------------

  analogReadResolution(12);

  // ----------------------------------------------------------
  // WIFI
  // ----------------------------------------------------------

  connectToWiFi();

  // ----------------------------------------------------------
  // DEVICE INFORMATION
  // ----------------------------------------------------------

  Serial.println();
  Serial.print("Device ID : ");
  Serial.println(DEVICE_ID);

  Serial.print("Backend   : ");
  Serial.println(BACKEND_URL);

  Serial.println();
  Serial.println("Starting sensor readings...");
  Serial.println();
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop() {

  // ----------------------------------------------------------
  // CHECK WIFI
  // ----------------------------------------------------------

  if (WiFi.status() != WL_CONNECTED) {

    Serial.println();
    Serial.println("WiFi disconnected!");

    connectToWiFi();
  }

  // ----------------------------------------------------------
  // FAST PATH: gas sensors + buzzer/LED every loop (~100ms)
  // Ensures 7s buzzer stop is never blocked by HTTP
  // ----------------------------------------------------------

  int mq135Raw = analogRead(MQ135_PIN);
  int mq2Digital = digitalRead(MQ2_DO_PIN);

  bool mq2GasDetected =
    (mq2Digital == MQ2_GAS_STATE);

  bool mq135GasDetected =
    (mq135Raw >= MQ135_THRESHOLD);

  updateAlarmOutputs(mq135GasDetected, mq2GasDetected);

  // ----------------------------------------------------------
  // SLOW PATH: DHT + Flask POST every 5 seconds
  // ----------------------------------------------------------

  if (millis() - lastSendTime >= SEND_INTERVAL_MS) {

    lastSendTime = millis();

    float temperature = dht.readTemperature();
    float humidity = dht.readHumidity();

    if (isnan(temperature) ||
        isnan(humidity)) {

      Serial.println(
        "ERROR: DHT11 reading failed!"
      );

      Serial.print("MQ135 Raw: ");
      Serial.println(mq135Raw);

      Serial.print("MQ2 DO: ");
      Serial.println(mq2Digital);

      // Do NOT return — keep buzzer 7s timer running
    } else {

      float gasValue =
        (mq135Raw / 4095.0) * 500.0;

      bool gasAlarm =
        mq135GasDetected ||
        mq2GasDetected;

      bool finalAlarm =
        gasAlarm ||
        aiHighRisk;

      // Send gasAlarm to backend (sensor semantics unchanged)
      sendSensorData(
        temperature,
        humidity,
        gasValue,
        mq135Raw,
        mq2Digital,
        gasAlarm
      );

      finalAlarm =
        mq135GasDetected ||
        mq2GasDetected ||
        aiHighRisk;

      // Update LCD: if within active scan result display window, show scan result, else show live telemetry
      if (millis() < scanResultShowUntil) {
        showLcdScanResult(lastAiScanClass, lastAiScanScore, lastAiScanIsSafe);
      } else {
        updateLcdTelemetry(temperature, humidity, (int)round(gasValue));
      }

      Serial.println("----------------------------------------");

      Serial.print("Temperature : ");
      Serial.print(temperature, 1);
      Serial.println(" °C");

      Serial.print("Humidity    : ");
      Serial.print(humidity, 1);
      Serial.println(" %");

      Serial.print("MQ135 Raw   : ");
      Serial.println(mq135Raw);

      Serial.print("MQ135 Index : ");
      Serial.println(gasValue, 1);

      Serial.print("MQ135 Alarm : ");
      Serial.println(mq135GasDetected ? "YES" : "NO");

      Serial.print("MQ2 DO      : ");
      Serial.println(mq2Digital);

      Serial.print("MQ2 Alarm   : ");
      Serial.println(mq2GasDetected ? "YES" : "NO");

      Serial.print("AI HIGH RISK: ");
      Serial.println(aiHighRisk ? "YES" : "NO");

      Serial.print("FINAL ALARM : ");
      Serial.println(finalAlarm ? "YES" : "NO");

      Serial.print("BUZZER      : ");
      Serial.println(buzzerSounding ? "ON" : "OFF");

      Serial.print("LED         : ");
      Serial.println(finalAlarm ? "ON" : "OFF");

      Serial.print("WiFi        : ");

      if (WiFi.status() == WL_CONNECTED) {
        Serial.println("CONNECTED");
      } else {
        Serial.println("DISCONNECTED");
      }

      Serial.println("----------------------------------------");
    }
  }

  delay(50);
}

// ============================================================
// WIFI CONNECTION
// ============================================================

void connectToWiFi() {

  Serial.println();
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);

  WiFi.begin(
    WIFI_SSID,
    WIFI_PASSWORD
  );

  int retries = 0;

  while (
    WiFi.status() != WL_CONNECTED &&
    retries < MAX_WIFI_RETRIES
  ) {

    delay(WIFI_RETRY_DELAY);

    Serial.print(".");

    retries++;
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {

    Serial.println("✅ WiFi Connected!");

    Serial.print("ESP32 IP Address: ");
    Serial.println(WiFi.localIP());

    Serial.print("WiFi Signal: ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");

  } else {

    Serial.println("❌ WiFi connection FAILED!");

    Serial.println(
      "ESP32 will retry automatically."
    );
  }
}

// ============================================================
// SEND DATA TO FLASK BACKEND
// ============================================================

void sendSensorData(
  float temperature,
  float humidity,
  float gas,
  int mq135Raw,
  int mq2Digital,
  bool alarm
) {

  // ----------------------------------------------------------
  // CHECK WIFI
  // ----------------------------------------------------------

  if (WiFi.status() != WL_CONNECTED) {

    Serial.println(
      "Cannot send data - WiFi disconnected."
    );

    return;
  }

  // ----------------------------------------------------------
  // HTTP CLIENT
  // ----------------------------------------------------------

  WiFiClient client;

  HTTPClient http;

  // ----------------------------------------------------------
  // START CONNECTION
  // ----------------------------------------------------------

  if (!http.begin(
        client,
        BACKEND_URL
      )) {

    Serial.println(
      "ERROR: Could not connect to backend."
    );

    return;
  }

  http.setTimeout(
    HTTP_TIMEOUT_MS
  );

  // ----------------------------------------------------------
  // HTTP HEADER
  // ----------------------------------------------------------

  http.addHeader(
    "Content-Type",
    "application/json"
  );

  // ----------------------------------------------------------
  // CREATE JSON
  // ----------------------------------------------------------

  StaticJsonDocument<512> doc;

  doc["device_id"] = DEVICE_ID;
  doc["source"] = "hardware";

  doc["temperature"] = temperature;

  doc["humidity"] = humidity;

  doc["gas"] = gas;

  doc["mq135_raw"] = mq135Raw;

  doc["mq2_digital"] = mq2Digital;

  doc["alarm"] = alarm;

  String jsonPayload;

  serializeJson(
    doc,
    jsonPayload
  );

  // ----------------------------------------------------------
  // SHOW DATA
  // ----------------------------------------------------------

  Serial.print("Sending: ");
  Serial.println(jsonPayload);

  // ----------------------------------------------------------
  // POST DATA
  // ----------------------------------------------------------

  int httpCode =
    http.POST(jsonPayload);

  // ----------------------------------------------------------
  // SERVER RESPONSE
  // ----------------------------------------------------------

  if (httpCode > 0) {

    Serial.print("HTTP Code: ");
    Serial.println(httpCode);

    String response =
      http.getString();

    Serial.print("Server Response: ");
    Serial.println(response);

    if (
      httpCode >= 200 &&
      httpCode < 300
    ) {

      Serial.println(
        "✅ SUCCESS: Sensor data sent!"
      );

      // Parse AI HIGH-RISK flag & alarm ID from Flask POST response
      StaticJsonDocument<512> respDoc;
      DeserializationError err =
        deserializeJson(respDoc, response);

      if (!err && respDoc.containsKey("ai_high_risk")) {
        bool newAiHighRisk = respDoc["ai_high_risk"].as<bool>();
        int newAlarmId = respDoc["ai_alarm_id"] | 0;

        // Check for NEW scan event (alarmId incremented or first scan result received)
        if (respDoc.containsKey("ai_top_prediction") && !respDoc["ai_top_prediction"].isNull()) {
          String newClass = respDoc["ai_top_prediction"].as<String>();
          float newScore = respDoc.containsKey("ai_final_score") ? respDoc["ai_final_score"].as<float>() : 0.0;
          bool newSafe = respDoc.containsKey("ai_is_safe") ? respDoc["ai_is_safe"].as<bool>() : !newAiHighRisk;

          if (newAlarmId != lastAiAlarmId && newClass.length() > 0) {
            lastAiScanClass = newClass;
            lastAiScanScore = newScore;
            lastAiScanIsSafe = newSafe;
            scanResultShowUntil = millis() + 15000; // Show scan result on LCD for 15 seconds!
            showLcdScanResult(lastAiScanClass, lastAiScanScore, lastAiScanIsSafe);
          }
        }

        // Trigger buzzer on NEW unsafe prediction event (even if previous alarm was latched)
        if (newAiHighRisk && (newAlarmId > lastAiAlarmId || !aiHighRisk)) {
          Serial.println("🚨 >>> NEW AI HIGH-RISK PREDICTION DETECTED! SOUNDING BUZZER! <<< 🚨");
          startBuzzer(BUZZER_DURATION_MS);
          digitalWrite(LED_PIN, HIGH);
          alarmWasActive = true;
        }

        aiHighRisk = newAiHighRisk;
        if (newAlarmId > lastAiAlarmId) {
          lastAiAlarmId = newAlarmId;
        }

        Serial.print("AI HIGH RISK from backend: ");
        Serial.print(aiHighRisk ? "YES" : "NO");
        Serial.print(" (Alarm ID: ");
        Serial.print(lastAiAlarmId);
        Serial.println(")");
      }

    } else {

      Serial.println(
        "❌ ERROR: Backend rejected request."
      );
    }

  } else {

    Serial.print(
      "HTTP Error: "
    );

    Serial.println(
      http.errorToString(httpCode)
    );
  }

  // ----------------------------------------------------------
  // CLOSE HTTP
  // ----------------------------------------------------------

  http.end();
}