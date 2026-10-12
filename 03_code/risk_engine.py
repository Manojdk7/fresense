from pathlib import Path
import joblib
import pandas as pd
from image_model import FoodImageModel

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "02_models"

# Food bacterial danger zone: 5°C – 60°C
DANGER_ZONE_LOW = 5.0
DANGER_ZONE_HIGH = 60.0

# ====================================================================
# Phase 2.1: Food Category-Specific Spoilage Profiles
# ====================================================================
FOOD_PROFILES = {
    "produce": {
        "name": "Produce (Fruits & Vegetables)",
        "key": "produce",
        "safe_temp_max": 25.0,       # Ambient storage (apples, bananas, tomatoes, citrus); heat stress > 25-30°C
        "extreme_temp_max": 38.0,
        "safe_humidity_max": 85.0,   # Natural humidity equilibrium for fresh produce
        "safe_gas_max": 200.0,       # Ethylene + ripening VOC threshold matching 200 index
        "visual_weight": 0.75,       # Visual mold, wrinkling, decay are primary (75%)
        "sensor_weight": 0.25,       # Environmental sensor support (25%)
        "danger_zone_concern": "Moderate (accelerated rotting and mold growth above 28-30°C)",
    },
    "meat": {
        "name": "Meat & Poultry",
        "key": "meat",
        "safe_temp_max": 4.0,         # Strict cold-chain; rapid pathogen growth (Salmonella/Listeria) >4°C
        "extreme_temp_max": 32.0,
        "safe_humidity_max": 70.0,
        "safe_gas_max": 60.0,         # Extreme sensitivity to ammonia (NH3), putrescine, and cadaverine
        "visual_weight": 0.40,
        "sensor_weight": 0.60,        # Gas & temp abuse signal spoilage long before surface discoloration
        "danger_zone_concern": "Critical (high risk of enteric pathogens)",
    },
    "dairy": {
        "name": "Dairy & Milk Products",
        "key": "dairy",
        "safe_temp_max": 4.0,         # Must be kept ≤ 4°C
        "extreme_temp_max": 32.0,
        "safe_humidity_max": 65.0,
        "safe_gas_max": 75.0,         # Lactic acidification and volatile sour organic acids
        "visual_weight": 0.45,
        "sensor_weight": 0.55,
        "danger_zone_concern": "High (rapid curdling and lactic acid fermentation)",
    },
    "bakery": {
        "name": "Bakery & Dry Goods",
        "key": "bakery",
        "safe_temp_max": 24.0,        # Safe at room temperature
        "extreme_temp_max": 45.0,
        "safe_humidity_max": 60.0,    # Extreme sensitivity to humidity; moisture triggers mold spores
        "safe_gas_max": 100.0,
        "visual_weight": 0.75,        # Surface mold is the definitive indicator
        "sensor_weight": 0.25,
        "danger_zone_concern": "Low bacterial, high mold growth under humidity",
    },
    "general": {
        "name": "General Food & Produce",
        "key": "general",
        "safe_temp_max": 25.0,
        "extreme_temp_max": 40.0,
        "safe_humidity_max": 80.0,
        "safe_gas_max": 200.0,
        "visual_weight": 0.75,
        "sensor_weight": 0.25,
        "danger_zone_concern": "Accelerated spoilage and decomposition above 30°C",
    },
}


def get_food_category(food_name_or_key: str) -> str:
    """Infer food category key ('produce', 'meat', 'dairy', 'bakery', 'general') from name."""
    if not food_name_or_key:
        return "general"
    s = str(food_name_or_key).lower().replace("_", " ")
    if any(k in s for k in ["chicken", "meat", "beef", "pork", "fish", "seafood", "poultry"]):
        return "meat"
    if any(k in s for k in ["milk", "cheese", "dairy", "yogurt", "butter"]):
        return "dairy"
    if any(k in s for k in ["bread", "bakery", "cake", "pastry", "flour", "grain"]):
        return "bakery"
    if any(k in s for k in [
        "apple", "banana", "tomato", "okra", "cucumber", "orange", "potato",
        "bitter", "capsicum", "carrot", "mango", "fruit", "vegetable", "spinach", "lettuce"
    ]):
        return "produce"
    return "general"


def _compute_temp_penalty(temperature: float, profile: dict = None) -> float:
    """Score 0–1 based on category-specific safe temperature threshold."""
    prof = profile or FOOD_PROFILES["general"]
    safe_max = prof["safe_temp_max"]
    extreme_max = prof.get("extreme_temp_max", 40.0)
    if temperature <= safe_max:
        return 0.0
    return min(1.0, (temperature - safe_max) / (extreme_max - safe_max))


def _compute_humidity_penalty(humidity: float, profile: dict = None) -> float:
    """Score 0–1 based on category-specific humidity threshold."""
    prof = profile or FOOD_PROFILES["general"]
    safe_max = prof["safe_humidity_max"]
    if humidity <= safe_max:
        return 0.0
    return min(1.0, (humidity - safe_max) / (100.0 - safe_max))


def _compute_gas_penalty(gas: float, profile: dict = None) -> float:
    """Score 0–1 based on category-specific gas/VOC threshold."""
    prof = profile or FOOD_PROFILES["general"]
    safe_max = prof["safe_gas_max"]
    extreme_max = 500.0
    if gas <= safe_max:
        return 0.0
    return min(1.0, (gas - safe_max) / (extreme_max - safe_max))


def _pick_primary_driver(
    visual: float, temp: float, humidity: float, gas: float
) -> str:
    """Return the name of whichever sub-score is highest."""
    drivers = {
        "Visual Spoilage": visual,
        "Temperature Abuse": temp,
        "Humidity Excess": humidity,
        "Gas / VOC Spike": gas,
    }
    return max(drivers, key=drivers.get)


def _make_recommendation(final_score: float, primary_driver: str, profile: dict = None) -> str:
    """Return a human-readable action recommendation informed by food category."""
    prof = profile or FOOD_PROFILES["general"]
    cat = prof.get("key", "general")

    if final_score < 0.35:
        return f"Safe for consumption ({prof['name']}). Cold chain intact."
    elif final_score < 0.60:
        if cat == "meat":
            return "Warning: High sensitivity for Meat/Poultry. Cook thoroughly today or discard."
        elif cat == "dairy":
            return "Warning: Dairy temperature/gas elevated. Check for sour odor or curdling."
        elif cat == "produce":
            return "Warning: Early ripening or moisture excess. Consume soon or trim affected areas."
        action_map = {
            "Visual Spoilage": "Inspect food surface carefully. Consume soon or cook thoroughly.",
            "Temperature Abuse": f"Reduce storage temperature below {prof['safe_temp_max']}°C immediately.",
            "Humidity Excess": "Improve storage ventilation to prevent mold propagation.",
            "Gas / VOC Spike": "Ventilate storage area. Check for early spoilage or gas accumulation.",
        }
        return action_map.get(primary_driver, "Monitor closely and reassess within 6 hours.")
    else:
        if cat == "meat":
            return "CRITICAL DISCARD: Meat spoilage or extreme temperature abuse detected. High pathogen risk."
        elif cat == "dairy":
            return "CRITICAL DISCARD: Dairy spoilage detected. Do not consume."
        if primary_driver == "Visual Spoilage":
            return "Discard immediately — visible spoilage or mold detected."
        elif primary_driver == "Temperature Abuse":
            return f"Discard — extended temperature abuse exceeds safe limit ({prof['safe_temp_max']}°C)."
        else:
            return "Discard or isolate — critical contamination risk detected."


def _explain_risk(
    visual_score: float,
    temperature: float,
    humidity: float,
    gas: float,
    final_score: float,
    profile: dict = None,
) -> dict:
    """Build full XAI explanation block for a risk result using category profile."""
    prof = profile or FOOD_PROFILES["general"]
    temp_penalty = _compute_temp_penalty(temperature, prof)
    humidity_penalty = _compute_humidity_penalty(humidity, prof)
    gas_penalty = _compute_gas_penalty(gas, prof)
    primary_driver = _pick_primary_driver(
        visual_score, temp_penalty, humidity_penalty, gas_penalty
    )
    recommendation = _make_recommendation(final_score, primary_driver, prof)

    # Determine risk tier label
    if final_score < 0.35:
        risk_level = "SAFE"
    elif final_score < 0.60:
        risk_level = "WARNING"
    else:
        risk_level = "UNSAFE"

    return {
        "risk_level": risk_level,
        "visual_score": round(float(visual_score), 4),
        "temp_penalty": round(float(temp_penalty), 4),
        "humidity_penalty": round(float(humidity_penalty), 4),
        "gas_penalty": round(float(gas_penalty), 4),
        "primary_driver": primary_driver,
        "recommendation": recommendation,
        "food_category": prof["key"],
        "category_name": prof["name"],
        "safe_temp_threshold": prof["safe_temp_max"],
        "safe_humidity_threshold": prof["safe_humidity_max"],
        "safe_gas_threshold": prof["safe_gas_max"],
    }


class RiskEngine:
    def __init__(self):
        print("Initializing Risk Engine...")
        self.image_model = FoodImageModel(
            model_path=str(MODELS_DIR / "food_spoilage_model_best.pth")
        )

        sensor_model_path = MODELS_DIR / "sensor_model.pkl"
        if sensor_model_path.exists():
            self.sensor_model = joblib.load(sensor_model_path)
            print("Sensor model loaded successfully!")
        else:
            raise FileNotFoundError(
                f"Sensor model not found at {sensor_model_path}"
            )

    def analyze(
        self,
        image_path,
        temperature,
        humidity,
        gas,
        mq135_raw=None,
        mq2_digital=None,
        alarm=None,
        category=None,
    ):
        # 1. Get image spoilage risk
        img_spoilage_score, top_pred = self.image_model.predict_spoilage(image_path)
        if top_pred.startswith("INVALID_NON_FOOD"):
            detail = top_pred.split(":", 1)[1] if ":" in top_pred else "Non-food item"
            raise ValueError(
                f"Non-food item detected ({detail}). Please upload an image of a supported food item."
            )
        if top_pred in {"ERROR_INVALID_IMAGE", "INVALID_IMAGE"}:
            raise ValueError("The uploaded file is not a valid image.")

        # Resolve category profile (user override or auto-inferred from prediction)
        cat_key = (category or "").lower().strip()
        if cat_key not in FOOD_PROFILES:
            cat_key = get_food_category(top_pred)
        profile = FOOD_PROFILES.get(cat_key, FOOD_PROFILES["general"])

        # 2. Get sensor spoilage risk
        sensor_data = pd.DataFrame(
            {
                "temperature": [temperature],
                "humidity": [humidity],
                "gas": [gas],
                "mq135_raw": [
                    mq135_raw if mq135_raw is not None else gas / 500.0 * 4095.0
                ],
                "mq2_digital": [mq2_digital if mq2_digital is not None else 1],
                "alarm": [int(bool(alarm)) if alarm is not None else 0],
            }
        )
        sensor_risk_score = float(self.sensor_model.predict(sensor_data)[0])
        sensor_risk_score = max(0.0, min(1.0, sensor_risk_score))

        # 3. Combine scores using category-specific weights
        vis_w = profile["visual_weight"]
        sens_w = profile["sensor_weight"]
        final_score = (img_spoilage_score * vis_w) + (sensor_risk_score * sens_w)
        final_score = max(0.0, min(1.0, final_score))

        # 4. Build XAI explanation with category profile
        explanation = _explain_risk(
            img_spoilage_score, temperature, humidity, gas, final_score, profile
        )

        return {
            "image_score": float(img_spoilage_score),
            "sensor_score": float(sensor_risk_score),
            "final_score": float(final_score),
            "top_prediction": top_pred,
            "is_safe": bool(final_score < 0.5),
            "is_food": True,
            # --- Phase 2: Category Profile fields ---
            "food_category": profile["key"],
            "category_name": profile["name"],
            "profile_applied": profile,
            # --- Phase 1: XAI fields ---
            "risk_level": explanation["risk_level"],
            "visual_score": explanation["visual_score"],
            "temp_penalty": explanation["temp_penalty"],
            "humidity_penalty": explanation["humidity_penalty"],
            "gas_penalty": explanation["gas_penalty"],
            "primary_driver": explanation["primary_driver"],
            "recommendation": explanation["recommendation"],
        }

    def analyze_sensors_only(
        self,
        temperature,
        humidity,
        gas,
        mq135_raw=None,
        mq2_digital=None,
        alarm=None,
        category=None,
    ):
        """Analyze food safety risk using sensor data and category profile (no image required)."""
        cat_key = (category or "general").lower().strip()
        profile = FOOD_PROFILES.get(cat_key, FOOD_PROFILES["general"])
        sensor_data = pd.DataFrame(
            {
                "temperature": [temperature],
                "humidity": [humidity],
                "gas": [gas],
                "mq135_raw": [
                    mq135_raw if mq135_raw is not None else gas / 500.0 * 4095.0
                ],
                "mq2_digital": [mq2_digital if mq2_digital is not None else 1],
                "alarm": [int(bool(alarm)) if alarm is not None else 0],
            }
        )
        sensor_risk_score = float(self.sensor_model.predict(sensor_data)[0])
        sensor_risk_score = max(0.0, min(1.0, sensor_risk_score))

        # For sensor-only mode, visual_score = 0 (no image)
        explanation = _explain_risk(
            0.0, temperature, humidity, gas, sensor_risk_score, profile
        )

        return {
            "sensor_score": float(sensor_risk_score),
            "final_score": float(sensor_risk_score),
            "is_safe": bool(sensor_risk_score < 0.5),
            "food_category": profile["key"],
            "category_name": profile["name"],
            "profile_applied": profile,
            # --- Phase 1: XAI fields ---
            "risk_level": explanation["risk_level"],
            "visual_score": 0.0,
            "temp_penalty": explanation["temp_penalty"],
            "humidity_penalty": explanation["humidity_penalty"],
            "gas_penalty": explanation["gas_penalty"],
            "primary_driver": explanation["primary_driver"],
            "recommendation": explanation["recommendation"],
        }


if __name__ == "__main__":
    import os
    engine = RiskEngine()
    test_image = "test.jpg"
    if os.path.exists(test_image):
        result = engine.analyze(test_image, 25.0, 60.0, 100.0)
        print("Test Result:", result)
    else:
        print(f"Could not find {test_image} to test. Please create one.")
