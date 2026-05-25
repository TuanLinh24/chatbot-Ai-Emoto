import json
import random

# Danh sách các hãng và tiền tố/hậu tố để sinh tên xe tự động
BRANDS = ["vinfast", "yadea", "pega", "dat_bike", "honda", "yamaha", "dibao", "sym"]
MODEL_PREFIXES = ["Alpha", "Beta", "Pro", "Lite", "Ultra", "Max", "S", "X", "V", "Evo"]
MODEL_NAMES = ["Feliz", "Impes", "Klara", "Vento", "Theon", "Weaver", "Gogo", "Pansy", "Od3", "Phantom"]

def generate_database(num_vehicles=100):
    db = {
        "thong_tin_chung": {
            "danh_sach_xe": "Hiện tại showroom chúng tôi phân phối chính hãng hơn 100 mẫu xe từ các thương hiệu: VinFast, Yadea, Pega, Dat Bike..."
        },
        "xe_may_dien": {}
    }

    # Tạo key cho các hãng
    for brand in BRANDS:
        db["xe_may_dien"][brand] = {}

    generated_count = 0
    while generated_count < num_vehicles:
        brand_key = random.choice(BRANDS)
        
        # Tạo tên xe ngẫu nhiên
        name_parts = []
        if random.random() > 0.3:
            name_parts.append(random.choice(MODEL_NAMES))
        name_parts.append(random.choice(MODEL_PREFIXES))
        
        model_name = " ".join(name_parts)
        model_key = model_name.lower().replace(" ", "_")
        
        # Đảm bảo không bị trùng key trong cùng 1 hãng
        if model_key not in db["xe_may_dien"][brand_key]:
            
            # Khởi tạo thông số ngẫu nhiên
            gia_ca = random.randint(15, 60) * 1000000
            cong_suat = random.choice([1000, 1500, 2000, 3000, 4000])
            toc_do = random.choice([50, 65, 78, 90, 100])
            quang_duong = random.choice([70, 90, 120, 150, 200])
            pin = random.choice(["Pin LFP", "Acquy Chì", "Pin Lithium-ion"])
            bh_xe = random.choice([2, 3, 4, 5])
            bh_pin = random.choice([3, 5, 10])

            brand_display = brand_key.replace("_", " ").title()
            if brand_key == "vinfast": brand_display = "VinFast"
            if brand_key == "dat_bike": brand_display = "Dat Bike"

            db["xe_may_dien"][brand_key][model_key] = {
                "ten_chuan": f"{brand_display} {model_name}",
                "attributes": {
                    "gia_ca": {
                        
                        "value": f"{gia_ca:,} VNĐ"
                    },
                    "thong_so": {
                        
                        "value": f"Động cơ {cong_suat}W, {pin}, Tốc độ đa {toc_do}km/h, Quãng đường {quang_duong}km/lần sạc."
                    },
                    "bao_hanh": {
                        
                        "value": f"Bảo hành chính hãng {bh_xe} năm đối với xe và {bh_pin} năm đối với pin."
                    }
                }
            }
            generated_count += 1

    with open("database_v2.json", "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=4)
    
    print(f"✅ Đã tạo thành công {generated_count} mẫu xe phân cấp vào file 'database_v2.json'!")

if __name__ == "__main__":
    generate_database(100)