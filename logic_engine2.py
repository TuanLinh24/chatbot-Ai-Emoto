import json
import time
import re
import numpy as np
import onnxruntime as ort
import requests
from tokenizers import Tokenizer  

class ChatbotEngine_V4:
    def __init__(self, db_path, intents_path="data/intents.json", onnx_model_path="models/model.onnx", tokenizer_path="data/tokenizer.json"):
        print("🤖 Khởi tạo Engine V4 [LAN Hybrid LLM & NumPy Vectorized Cache Ready]...")
        
        self.db_path = db_path
        self.intents_path = intents_path
        self.onnx_model_path = onnx_model_path
        self.tokenizer_path = tokenizer_path
        
        # 🏎️ TỐI ƯU HÓA ONNX RUNTIME DÀNH RIÊNG CHO RASPBERRY PI 4 (4 CORES)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4  # Khai thác toàn bộ 4 nhân physical của Pi 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.ort_session = ort.InferenceSession(onnx_model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Cấu hình bộ nhớ đệm Cache
        self.l1_cache = {}       
        self.l2_cache = []       
        
        self.l2_threshold = 0.88       
        self.global_intent_threshold = 0.68  
        self.static_intent_threshold = 0.72  
        self.max_l2_size = 300
        
        # 🎯 BỘ LỌC TỪ KHÓA BẢO VỆ INTENT CHÍNH (Chỉ dùng để bổ trợ lọc nhiễu diện hẹp)
        self.intent_keywords = {
            "gia_ca": ["giá", "nhiêu", "tiền", "phí", "bảng giá", "mua", "niêm yết", "lăn bánh", "triệu", "vnd"],
            "thong_so": ["thông số", "vận tốc", "tốc độ", "pin", "acquy", "ắc quy", "km", "sạc", "watt", "công suất", "động cơ", "cao", "nặng", "cốp", "xa", "dài", "rộng"],
            "bao_hanh": ["bảo hành", "sửa", "lỗi", "bảo dưỡng", "định kỳ", "trạm", "lâu", "hỏng", "bảo trì"]
        }
        
        self.pc_llm_url = "http://192.168.1.50:11434/v1/chat"  
        self.reload_database()
        print("✅ Hệ thống Hybrid V4 kết hợp Màng lọc Ngữ nghĩa sẵn sàng!")

    def _normalize_for_cache(self, text):
        """CHUẨN HÓA ĐỒNG BỘ: Xóa dấu câu, chuẩn hóa khoảng trắng, đưa về ký tự thường"""
        text = re.sub(r'[^\w\sÀ-ỹ]', ' ', text)  # Giữ lại toàn bộ dải ký tự tiếng Việt có dấu
        return " ".join(text.lower().strip().split())

    def reload_database(self):
        start_reload = time.time()
        print("🔄 [V4] Đang nạp cấu hình và tiến hành Vector hóa ngữ nghĩa...")
        
        with open(self.db_path, "r", encoding="utf-8") as f:
            self.db = json.load(f)
        with open(self.intents_path, "r", encoding="utf-8") as f:
            self.intents_config = json.load(f)
            
        self.l1_cache.clear()
        self.l2_cache.clear()

        # 1. Vector hóa Ý định (Intents Matrix)
        self.intent_matrices = {}
        self.global_intent_labels = {
          "gia_ca": ["giá bán chính thức bao nhiêu tiền", "chi phí lăn bánh", "mua xe hết bao nhiêu", "bảng giá xe máy điện", "giá niêm yết chính thức"],
          "thong_so": ["thông số kỹ thuật chi tiết", "vận tốc tối đa đạt bao nhiêu", "dung lượng pin chạy được bao nhiêu km", "công suất động cơ bao nhiêu watt"],
          "bao_hanh": ["chính sách bảo hành mấy năm", "sửa chữa xe gặp lỗi", "lịch bảo dưỡng định kỳ", "bảo hành pin như thế nào"]
        }
        
        for intent_key, phrases in self.global_intent_labels.items():
            vectors = [self._get_embedding(self._normalize_for_cache(phrase)) for phrase in phrases]
            self.intent_matrices[intent_key] = np.vstack(vectors)
            
        # 2. Xây dựng bộ từ điển thực thể phân cấp (Tránh lỗi \b Tiếng Việt bằng Lookarounds)
        self.brands = {}
        if "xe_may_dien" in self.db:
            for brand_key, models in self.db["xe_may_dien"].items():
                brand_names = [brand_key.replace("_", " ")]
                if brand_key == "vinfast": brand_names.extend(["vin phát", "vin fast", "vf"])
                
                model_list = []
                for model_key, model_data in models.items():
                    ten_chuan = model_data.get("ten_chuan", "").lower()
                    clean_model_name = ten_chuan.replace(brand_key.replace("_", " "), "").strip()
                    
                    search_terms = [model_key.replace("_", " ")]
                    if clean_model_name and clean_model_name != brand_key.replace("_", " "):
                        search_terms.insert(0, clean_model_name)
                        
                    model_list.append({
                        "model_key": model_key,
                        "search_terms": sorted(list(set(search_terms)), key=len, reverse=True)
                    })
                
                self.brands[brand_key] = {
                    "search_terms": brand_names,
                    "models": sorted(model_list, key=lambda x: max(len(t) for t in x["search_terms"]), reverse=True)
                }

        # 🎯 3. MÀNG LỌC NGỮ NGHĨA TƯƠNG PHẢN (Dùng thay thế hoàn toàn cho lọc từ khóa thô)
        # Tập Anchor Hợp lệ (In-Domain)
        self.ev_anchor_phrases = [
            "sạc pin xe máy điện ở đâu", "xe điện đi trời mưa có sao không", "tuổi thọ pin lfp bao lâu", 
            "hết điện giữa đường thì làm sao", "sạc qua đêm có cháy nổ không", "so sánh pin lithium và ắc quy chì",
            "chi phí sạc điện một tháng bao nhiêu", "bảo dưỡng xe máy điện gồm những gì", "xe máy điện đăng ký biển số"
        ]
        ev_vectors = [self._get_embedding(self._normalize_for_cache(phrase)) for phrase in self.ev_anchor_phrases]
        self.ev_domain_matrix = np.vstack(ev_vectors)

        # Tập Anchor Lạc đề / Rác (Out-of-Domain) - Dùng ngữ nghĩa để bắt trọn mọi biến thể vô nghĩa
        self.out_of_scope_phrases = [
            "hướng dẫn nấu ăn ngon tại nhà", "cách nấu cơm nấu phở luộc gà", "nghe bài hát mới của ca sĩ",
            "lên kế hoạch đi đá bóng đá banh", "giải bài toán hình học đại số", "thời tiết hôm nay thế nào",
            "chơi game giải trí trực tuyến", "làm bài thơ tình lãng mạn", "tâm sự chuyện tình cảm cá nhân"
        ]
        out_vectors = [self._get_embedding(self._normalize_for_cache(phrase)) for phrase in self.out_of_scope_phrases]
        self.out_of_scope_matrix = np.vstack(out_vectors)

        print(f"⚡ Đã tối ưu xong không gian Vector! Thời gian: {round(time.time() - start_reload, 3)} giây!")

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
        """Sử dụng Lookarounds (?<!\w) để loại bỏ hoàn toàn lỗi nhận diện sai ký tự Tiếng Việt của \\b"""
        detected_brand = None
        for brand_key, brand_data in self.brands.items():
            for term in brand_data["search_terms"]:
                pattern = r'(?<![a-zA-Z0-9_À-ỹ])' + re.escape(term) + r'(?![a-zA-Z0-9_À-ỹ])'
                if re.search(pattern, clean_text):
                    detected_brand = brand_key
                    break
            if detected_brand: break
            
        if detected_brand:
            for model in self.brands[detected_brand]["models"]:
                for term in model["search_terms"]:
                    if term:
                        pattern = r'(?<![a-zA-Z0-9_À-ỹ])' + re.escape(term) + r'(?![a-zA-Z0-9_À-ỹ])'
                        if re.search(pattern, clean_text):
                            return (detected_brand, model["model_key"])
        else:
            for brand_key, brand_data in self.brands.items():
                for model in brand_data["models"]:
                    for term in model["search_terms"]:
                        if term:
                            pattern = r'(?<![a-zA-Z0-9_À-ỹ])' + re.escape(term) + r'(?![a-zA-Z0-9_À-ỹ])'
                            if re.search(pattern, clean_text):
                                return (brand_key, model["model_key"])
                            
        return (detected_brand, None) if detected_brand else None

    def _is_ev_related(self, query_vector, clean_query):
        """BỘ LỌC TƯƠNG PHẢN MA TRẬN: Xác định xem câu hỏi có thực sự thuộc chủ đề xe điện không"""
        # Tính toán độ tương đồng đồng thời với tập Hợp lệ và Lạc đề
        in_domain_sim = np.max(np.dot(self.ev_domain_matrix, query_vector.T).flatten())
        out_domain_sim = np.max(np.dot(self.out_of_scope_matrix, query_vector.T).flatten())
        
        # Nếu điểm số Vector nghiêng hẳn về chủ đề ngoài ngành hoặc điểm số ngành quá thấp -> Chặn đứng
        if out_domain_sim > in_domain_sim and out_domain_sim > 0.50:
            return False
        
        if in_domain_sim >= 0.52:  
            return True
            
        return False

    def _call_pc_llm_via_lan(self, user_message):
        headers = {"Content-Type": "application/json"}
        payload = {"message": user_message}
        try:
            response = requests.post(self.pc_llm_url, json=payload, headers=headers, timeout=12.0)
            if response.status_code == 200:
                res_json = response.json()
                if res_json.get("status") == "success":
                    return res_json.get("reply", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Thất bại khi kết nối tới LLM Server qua mạng LAN: {e}")
        return None

    def get_response(self, user_query):
        start_time = time.time()
        clean_query = self._normalize_for_cache(user_query)
        
        if not clean_query:
            return "Dạ em chào anh/chị, anh/chị cần hỗ trợ gì về xe máy điện không ạ?", round(time.time() - start_time, 4), "EMPTY_QUERY"
            
        if clean_query in self.l1_cache:
            return self.l1_cache[clean_query], round(time.time() - start_time, 4), "L1_HIT"
            
        current_entity = self._extract_entity(clean_query)
        query_vector = self._get_embedding(clean_query)

        # 🛡️ KIỂM TRA ĐỘ LẠC ĐỀ NGAY TẠI EDGE BẰNG MA TRẬN TƯƠNG PHẢN
        if not self._is_ev_related(query_vector, clean_query) and not current_entity:
            fallback_msg = "Dạ, hiện tại em chưa hiểu rõ ý của anh/chị. Anh/chị có thể hỏi cụ thể hơn về Giá cả, Thông số kỹ thuật hoặc Bảo hành của các dòng xe máy điện được không ạ?"
            self.l1_cache[clean_query] = fallback_msg
            return fallback_msg, round(time.time() - start_time, 4), "SEMANTIC_REJECT"

        # Định tuyến Ý định (Intent Routing)
        current_intent = None
        intent_scores = {}
        for intent_key, matrix in self.intent_matrices.items():
            similarities = np.dot(matrix, query_vector.T).flatten()
            intent_scores[intent_key] = np.max(similarities)
            
        if intent_scores:
            best_intent, max_intent_score = max(intent_scores.items(), key=lambda x: x[1])
            if max_intent_score >= self.global_intent_threshold:
                if best_intent in self.intent_keywords:
                    if any(kw in clean_query for kw in self.intent_keywords[best_intent]):
                        current_intent = best_intent
                else:
                    current_intent = best_intent

        # TẦNG 2: CACHE L2 
        best_l2_match = None
        max_l2_similarity = -1
        matching_indices = [
            i for i, cached_item in enumerate(self.l2_cache) 
            if cached_item.get("entity") == current_entity and cached_item.get("intent") == current_intent
        ]
        
        if matching_indices:
            cached_vectors = np.vstack([self.l2_cache[i]["vector"] for i in matching_indices])
            similarities = np.dot(cached_vectors, query_vector.T).flatten()
            max_idx = np.argmax(similarities)
            if similarities[max_idx] >= self.l2_threshold:
                max_l2_similarity = similarities[max_idx]
                best_l2_match = self.l2_cache[matching_indices[max_idx]]
                
        if best_l2_match and max_l2_similarity >= self.l2_threshold:
            final_reply = best_l2_match["reply"]
            self.l1_cache[clean_query] = final_reply 
            return final_reply, round(time.time() - start_time, 4), f"L2_HIT ({round(max_l2_similarity * 100, 1)}%)"

        # TẦNG 3: TRUY XUẤT CƠ SỞ DỮ LIỆU GỐC CỤC BỘ
        final_reply = None
        cache_status = "CACHE_MISS"
        
        if current_intent in ["gia_ca", "thong_so", "bao_hanh"]:
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
                has_explicit_keyword = any(kw in clean_query for kw in self.intent_keywords.get(current_intent, []))
                if has_explicit_keyword:
                    intent_names = {"gia_ca": "giá bán", "thong_so": "thông số kỹ thuật", "bao_hanh": "chính sách bảo hành"}
                    final_reply = f"Dạ, em hiểu anh/chị cần hỏi về {intent_names[current_intent]}, nhưng anh/chị đang quan tâm cụ thể hãng hay mẫu xe nào ạ (VD: VinFast, Yadea, Dat Bike...)?"

        # 🌐 TẦNG 4: CHUYỂN TIẾP SANG PC FLASK LLM SERVER
        if not final_reply:
            print("🌐 Gửi yêu cầu phân tích chuyên sâu qua mạng LAN tới PC LLM...")
            llm_reply = self._call_pc_llm_via_lan(user_query)
            if llm_reply:
                if "OUT_OF_SCOPE_TRIGGERED" in llm_reply:
                    final_reply = "Dạ, hiện tại em chưa hiểu rõ ý của anh/chị. Anh/chị có thể hỏi cụ thể hơn về các vấn đề kỹ thuật hoặc thông tin xe máy điện được không ạ?"
                    cache_status = "LLM_GUARD_REJECT"
                else:
                    final_reply = llm_reply
                    cache_status = "PC_LLM_HIT"

        if not final_reply:
            final_reply = "Dạ, hiện tại em chưa hiểu rõ ý của anh/chị hoặc thông tin này nằm ngoài cơ sở dữ liệu. Anh/chị có thể hỏi cụ thể hơn về Giá cả, Thông số hoặc Bảo hành của các dòng xe máy điện được không ạ?"

        # Đồng bộ ngược vào Bộ nhớ đệm Cache
        self.l1_cache[clean_query] = final_reply
        self.l2_cache.append({
            "vector": query_vector,
            "reply": final_reply,
            "entity": current_entity,
            "intent": current_intent
        })
        if len(self.l2_cache) > self.max_l2_size:
            self.l2_cache.pop(0)

        return final_reply, round(time.time() - start_time, 4), cache_status