import json
import time
import random
import re
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer  

class ChatbotEngine_V4:
    def __init__(self, db_path, intents_path="intents.json", onnx_model_path="models/model.onnx", tokenizer_path="data/tokenizer.json"):
        print("🤖 Khởi tạo Engine V4 [Pi 4 Highly Optimized]: Regex + Hierarchical Entity + Hardcoded Fallback...")
        
        self.db_path = db_path
        self.intents_path = intents_path
        
        # Không dùng LLM nữa, chỉ dùng ONNX cho NLU để tối đa hóa tốc độ trên Pi 4
        self.ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        self.l1_cache = {}       
        self.l2_cache = []       
        
        self.l2_threshold = 0.86       
        self.global_intent_threshold = 0.62  
        self.static_intent_threshold = 0.72  
        self.max_l2_size = 250
        
        self.reload_database()
        print("✅ Hệ thống V4 sẵn sàng!")

    def _normalize_for_cache(self, text):
        """CHUẨN HÓA TRIỆT ĐỂ: Xóa toàn bộ dấu câu, ký tự đặc biệt, đưa về chữ thường"""
        # Xóa các ký tự không phải là chữ cái, số, hoặc khoảng trắng
        text = re.sub(r'[^\w\s]', ' ', text)
        return " ".join(text.lower().strip().split())

    def reload_database(self):
        start_reload = time.time()
        print("🔄 [V4] Đang nạp cấu hình và tiến hành Vector hóa...")
        
        with open(self.db_path, "r", encoding="utf-8") as f:
            self.db = json.load(f)
        with open(self.intents_path, "r", encoding="utf-8") as f:
            self.intents_config = json.load(f)
            
        self.l1_cache.clear()
        self.l2_cache.clear()

        # Vector hóa Intent (Giữ nguyên logic của bạn)
        self.intent_matrices = {}
        for intent, data in self.intents_config.items():
            samples = data.get("samples", [])
            if samples:
                vectors = [self._get_embedding(self._normalize_for_cache(phrase)) for phrase in samples]
                self.intent_matrices[intent] = np.vstack(vectors)

        self.global_intent_labels = {
          "gia_ca": ["giá bán chính thức bao nhiêu tiền", "chi phí lăn bánh", "mua xe hết bao nhiêu", "bảng giá xe máy điện", "giá niêm yết chính thức", "mua xe này bao nhiêu tiền", "giá thế nào"],
          "thong_so": ["thông số kỹ thuật chi tiết", "vận tốc tối đa đạt bao nhiêu", "dung lượng pin chạy được bao nhiêu km", "công suất động cơ bao nhiêu watt", "kích thước chiều cao cân nặng cốp xe", "xe này đi được xa không", "tốc độ tối đa", "pin chạy được bao nhiêu cây số", "thông số"],
          "bao_hanh": ["chính sách bảo hành mấy năm", "sửa chữa xe gặp lỗi", "lịch bảo dưỡng định kỳ", "bảo hành pin như thế nào", "trạm bảo hành sửa chữa ở đâu", "xe được bảo hành bao lâu"]
        }
        
        for intent_key, phrases in self.global_intent_labels.items():
            if phrases:
                vectors = [self._get_embedding(self._normalize_for_cache(phrase)) for phrase in phrases]
                self.intent_matrices[intent_key] = np.vstack(vectors)
            
        # XÂY DỰNG TỪ ĐIỂN THỰC THỂ PHÂN CẤP (Brand -> Models)
        self.brands = {}
        if "xe_may_dien" in self.db:
            for brand_key, models in self.db["xe_may_dien"].items():
                brand_names = [brand_key.replace("_", " ")]
                # Thêm biến thể cho hãng (ví dụ "vin phát" -> "vinfast")
                if brand_key == "vinfast": brand_names.extend(["vin phát", "vin fast", "vf"])
                
                model_list = []
                for model_key, model_data in models.items():
                    ten_chuan = model_data.get("ten_chuan", "").lower()
                    # Tách tên xe khỏi tên hãng để tìm kiếm độc lập (VD: "VinFast Alpha" -> "alpha")
                    clean_model_name = ten_chuan.replace(brand_key.replace("_", " "), "").strip()
                    model_list.append({
                        "model_key": model_key,
                        "search_terms": [clean_model_name, model_key.replace("_", " ")]
                    })
                
                self.brands[brand_key] = {
                    "search_terms": brand_names,
                    "models": sorted(model_list, key=lambda x: len(x["search_terms"][0]), reverse=True)
                }

        print(f"⚡ Đã tối ưu xong! Thời gian: {round(time.time() - start_reload, 3)} giây!")

    def _get_embedding(self, text):
        full_text = f"query: {text}"
        encoded = self.tokenizer.encode(full_text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        
        onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        input_names = [x.name for x in self.ort_session.get_inputs()]
        if "token_type_ids" in input_names:
            onnx_inputs["token_type_ids"] = np.array([encoded.type_ids], dtype=np.int64)
            
        outputs = self.ort_session.run(None, onnx_inputs)
        last_hidden_state = outputs[0]
        
        input_mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
        sum_mask = np.maximum(input_mask_expanded.sum(axis=1), 1e-9)
        embedding = sum_embeddings / sum_mask
        
        norm = np.linalg.norm(embedding)
        return embedding / (norm if norm > 0 else 1e-9)

    def _extract_entity(self, clean_text):
        """TRÍCH XUẤT PHÂN CẤP: Tìm Hãng trước -> Tìm Xe sau"""
        detected_brand = None
        
        # 1. Tìm Hãng xe
        for brand_key, brand_data in self.brands.items():
            for term in brand_data["search_terms"]:
                if term in clean_text:
                    detected_brand = brand_key
                    break
            if detected_brand: break
            
        # 2. Tìm Dòng xe
        detected_model = None
        
        # Nếu đã biết hãng, CHỈ tìm xe trong hãng đó (Ngăn lỗi "VinFast Alpha" thành "Yadea Alpha")
        if detected_brand:
            for model in self.brands[detected_brand]["models"]:
                for term in model["search_terms"]:
                    if term and term in clean_text:
                        return (detected_brand, model["model_key"])
        
        # Nếu chưa biết hãng, tìm trên toàn bộ cơ sở dữ liệu
        else:
            for brand_key, brand_data in self.brands.items():
                for model in brand_data["models"]:
                    for term in model["search_terms"]:
                        if term and term in clean_text:
                            # Trả về xe đầu tiên tìm thấy
                            return (brand_key, model["model_key"])
                            
        return (detected_brand, None) if detected_brand else None

    def get_response(self, user_query):
        start_time = time.time()
        
        # CHUẨN HÓA VĂN BẢN TRƯỚC TIÊN
        clean_query = self._normalize_for_cache(user_query)
        if not clean_query:
            return "Dạ em chào anh/chị, anh/chị cần hỗ trợ gì ạ?", round(time.time() - start_time, 4), "EMPTY_QUERY"
            
        cache_key = clean_query
        
        # TẦNG 1: CACHE L1
        if cache_key in self.l1_cache:
            return self.l1_cache[cache_key], round(time.time() - start_time, 4), "L1_HIT"
            
        # Sử dụng query ĐÃ CHUẨN HÓA cho NLU và L2
        current_entity = self._extract_entity(clean_query)
        query_vector = self._get_embedding(clean_query)

        # Định tuyến Ngữ nghĩa
        current_intent = None
        intent_scores = {}
        for intent_key, matrix in self.intent_matrices.items():
            similarities = np.dot(matrix, query_vector.T).flatten()
            intent_scores[intent_key] = np.max(similarities)
            
        if intent_scores:
            best_intent, max_intent_score = max(intent_scores.items(), key=lambda x: x[1])
            is_static = best_intent in self.intents_config
            threshold = self.static_intent_threshold if is_static else self.global_intent_threshold
            if max_intent_score >= threshold:
                current_intent = best_intent

        # TẦNG 2: CACHE L2
        best_l2_match = None
        max_l2_similarity = -1
        for cached_item in self.l2_cache:
            if cached_item.get("entity") == current_entity and cached_item.get("intent") == current_intent:
                similarity = np.dot(query_vector, cached_item["vector"].T)[0][0]
                if similarity > max_l2_similarity:
                    max_l2_similarity = similarity
                    best_l2_match = cached_item
                
        if max_l2_similarity >= self.l2_threshold:
            final_reply = best_l2_match["reply"]
            self.l1_cache[cache_key] = final_reply 
            return final_reply, round(time.time() - start_time, 4), f"L2_HIT ({round(max_l2_similarity * 100, 1)}%)"

        # TẦNG 3: ĐÁP ỨNG TRUY XUẤT
        final_reply = None
        
        if current_intent:
            if current_intent in self.intents_config:
                if current_intent == "chao_hoi":
                    final_reply = self.intents_config["chao_hoi"]["response_template"]
                elif current_intent == "danh_sach_xe":
                    final_reply = self.db.get("thong_tin_chung", {}).get("danh_sach_xe", "Dạ hiện tại bên em có rất nhiều dòng xe.")
            
            elif current_intent in ["gia_ca", "thong_so", "bao_hanh"]:
                # current_entity lúc này là một tuple: (brand_key, model_key)
                if current_entity and isinstance(current_entity, tuple) and current_entity[1]:
                    brand_key, model_key = current_entity
                    entity_data = self.db.get("xe_may_dien", {}).get(brand_key, {}).get(model_key)
                    
                    if entity_data:
                        ten_xe = entity_data.get("ten_chuan", "")
                        attrs = entity_data.get("attributes", {})
                        
                        if current_intent == "gia_ca" and "gia_ca" in attrs:
                            final_reply = f"Dạ, xe {ten_xe} hiện có giá niêm yết là {attrs['gia_ca']['value']} ạ."
                        elif current_intent == "thong_so" and "thong_so" in attrs:
                            final_reply = f"Dạ, thông số của {ten_xe} như sau: {attrs['thong_so']['value']}"
                        elif current_intent == "bao_hanh" and "bao_hanh" in attrs:
                            final_reply = f"Dạ, chính sách của {ten_xe}: {attrs['bao_hanh']['value']}"
                
                else:
                    intent_names = {"gia_ca": "giá bán", "thong_so": "thông số kỹ thuật", "bao_hanh": "chính sách bảo hành"}
                    final_reply = f"Dạ, em hiểu anh/chị cần hỏi về {intent_names[current_intent]}, nhưng anh/chị đang quan tâm cụ thể hãng hay mẫu xe nào ạ (VD: VinFast, Yadea, Dat Bike...)?"

        # TẦNG 4: HARDCODED FALLBACK (Thay thế LLM)
        if not final_reply:
            final_reply = "Dạ, hiện tại em chưa hiểu rõ ý của anh/chị hoặc thông tin này nằm ngoài cơ sở dữ liệu. Anh/chị có thể hỏi cụ thể hơn về Giá cả, Thông số hoặc Bảo hành của các dòng xe máy điện được không ạ?"

        self.l1_cache[cache_key] = final_reply
        self.l2_cache.append({
            "vector": query_vector,
            "reply": final_reply,
            "entity": current_entity,
            "intent": current_intent
        })
        
        if len(self.l2_cache) > self.max_l2_size:
            self.l2_cache.pop(0)

        return final_reply, round(time.time() - start_time, 4), "CACHE_MISS"