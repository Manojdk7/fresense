import requests
import io
from PIL import Image

def test_phase2():
    print("=== Testing Category Profiles ===")
    r1 = requests.post('http://localhost:5000/api/sensors/analyze', json={'temperature': 10, 'humidity': 65, 'gas': 45, 'category': 'produce'}).json()
    r2 = requests.post('http://localhost:5000/api/sensors/analyze', json={'temperature': 10, 'humidity': 65, 'gas': 45, 'category': 'meat'}).json()

    print(f"Produce at 10C -> Risk: {r1['risk_level']}, Temp Penalty: {r1['temp_penalty']}, Profile: {r1['category_name']}")
    print(f"Meat at 10C    -> Risk: {r2['risk_level']}, Temp Penalty: {r2['temp_penalty']}, Profile: {r2['category_name']}")
    assert r2['temp_penalty'] > r1['temp_penalty'], "Meat should have stricter temperature penalty than produce!"

    print("\n=== Testing Real Food Scan with Category Profile ===")
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, '..', '01_datasets', 'images', 'fresh_rotten_classification', 'dataset', 'Test', 'freshapples')
    img_name = os.listdir(img_dir)[0]
    img_path = os.path.join(img_dir, img_name)
    with open(img_path, 'rb') as f:
        files = {'image': ('apple.png', f, 'image/png')}
        data = {'temperature': '4', 'humidity': '60', 'gas': '40', 'category': 'produce'}
        r3 = requests.post('http://localhost:5000/analyze', files=files, data=data)
    print('Food image upload status:', r3.status_code)
    d3 = r3.json()
    print(f"Top Pred: {d3.get('top_prediction')}, Risk: {d3.get('risk_level')}, Profile: {d3.get('category_name')}")
    assert r3.status_code == 200, "Food image scan should return 200"
    assert d3.get('is_food') is True, "Food image should be recognized as food"

    print("\n=== Testing Non-Food Image (OOD Guardrail) ===")
    # Uniform gray image with zero texture
    noise_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    noise_img.save(buf, format='JPEG')
    buf.seek(0)
    files = {'image': ('non_food_gray.jpg', buf, 'image/jpeg')}
    data = {'temperature': '25', 'humidity': '50', 'gas': '40'}
    r4 = requests.post('http://localhost:5000/analyze', files=files, data=data)
    print('Non-food status code:', r4.status_code)
    d4 = r4.json()
    print('Non-food response:', d4)
    assert r4.status_code == 422, "OOD non-food image should return HTTP 422"
    assert d4.get('is_food') is False, "is_food should be False"

    print("\n=== Testing Model Benchmark Metrics API ===")
    r5 = requests.get('http://localhost:5000/api/model/metrics')
    assert r5.status_code == 200, "Metrics API should return 200"
    m = r5.json().get('metrics', {})
    print(f"Model: {m.get('model_name')}")
    print(f"Multiclass Accuracy: {m.get('overall_multiclass_accuracy')}%")
    print(f"Binary Freshness F1: {m.get('binary_freshness', {}).get('f1_score')}")
    print(f"Test Samples: {m.get('total_test_samples_evaluated')}")

    print("\nALL PHASE 2 VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    test_phase2()
