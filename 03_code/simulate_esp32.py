"""
FoodSafe-IoT — ESP32 Sensor Telemetry Simulator
Simulates periodic hardware telemetry from an ESP32 microcontroller
running DHT11, MQ-135, and MQ-2 sensors into the Flask backend.

Usage:
    python simulate_esp32.py               # Ambient-temperature demo (every 5s)
  python simulate_esp32.py --warm        # Simulate warming above 4°C into danger zone
  python simulate_esp32.py --spoil       # Simulate spoilage gas & VOC spike
  python simulate_esp32.py --once        # Send a single reading and exit
"""

import argparse
import json
import random
import sys
import time
import urllib.request
import urllib.error


def send_reading(url, device_id, temp, hum, gas, mq135_raw, mq2_digital, alarm):
    payload = {
        "device_id": device_id,
        "source": "simulation",
        "temperature": round(temp, 1),
        "humidity": round(hum, 1),
        "gas": round(gas, 1),
        "mq135_raw": int(mq135_raw),
        "mq2_digital": int(mq2_digital),
        "alarm": bool(alarm),
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return True, data
    except urllib.error.URLError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="FoodSafe-IoT ESP32 Sensor Simulator")
    parser.add_argument("--url", default="http://127.0.0.1:5000/api/sensors/data", help="Backend ingestion URL")
    parser.add_argument("--device", default="SIM_ESP32_001", help="Device ID")
    parser.add_argument("--interval", type=float, default=5.0, help="Interval between telemetry packets (seconds)")
    parser.add_argument("--warm", action="store_true", help="Simulate temperature warming (danger zone: 12-24°C)")
    parser.add_argument("--spoil", action="store_true", help="Simulate spoiled food gas/VOC surge")
    parser.add_argument("--once", action="store_true", help="Send a single packet and exit")
    parser.add_argument("--count", type=int, default=0, help="Number of packets to send (0 = infinite)")

    args = parser.parse_args()

    print("============================================================")
    print(" FoodSafe-IoT — ESP32 Sensor Telemetry Simulator")
    print("============================================================")
    print(f" Target URL : {args.url}")
    print(f" Device ID  : {args.device}")
    print(f" Mode       : {'Spoilage Spike' if args.spoil else ('Danger Zone Warming' if args.warm else 'Ambient Temperature Demo')}")
    print(f" Interval   : {args.interval}s")
    print(" Press Ctrl+C to stop simulation.\n")

    sent = 0
    while True:
        sent += 1
        if args.spoil:
            # Spoilage scenario: high temp, moisture, high ammonia and VOC
            temp = random.uniform(18.0, 26.0)
            hum = random.uniform(75.0, 92.0)
            gas = random.uniform(120.0, 240.0)
            mq135_raw = random.randint(620, 980)
            mq2_digital = 0 if random.random() < 0.6 else 1
            alarm = True
        elif args.warm:
            # Warming scenario: temperature drifting into danger zone
            temp = random.uniform(8.5, 18.0)
            hum = random.uniform(60.0, 74.0)
            gas = random.uniform(45.0, 75.0)
            mq135_raw = random.randint(380, 520)
            mq2_digital = 1
            alarm = False
        else:
            # Room-temperature ambient reading for the default demo
            temp = random.uniform(27.0, 28.0)
            hum = random.uniform(58.0, 66.0)
            gas = random.uniform(32.0, 48.0)
            mq135_raw = random.randint(310, 390)
            mq2_digital = 1
            alarm = False

        timestamp = time.strftime("%H:%M:%S")
        ok, res = send_reading(args.url, args.device, temp, hum, gas, mq135_raw, mq2_digital, alarm)

        if ok:
            print(
                f"[{timestamp}] Packet #{sent:03d} -> Temp: {temp:.1f}°C | "
                f"Hum: {hum:.1f}% | Gas: {gas:.1f} ppm | VOC: {mq135_raw} | "
                f"Status: OK (ai_high_risk={res.get('ai_high_risk')})"
            )
        else:
            print(f"[{timestamp}] Packet #{sent:03d} -> FAILED to send: {res}")

        if args.once:
            break
        if args.count > 0 and sent >= args.count:
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        sys.exit(0)
