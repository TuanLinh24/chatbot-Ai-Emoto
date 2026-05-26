import json
import time
import re
import numpy as np
import onnxruntime as ort
import requests
from tokenizers import Tokenizer  

class ChatbotEngine_V4:
    def __init__(self, db_path="database_v2.json", intents_path="data/intents.json", onnx_model_path="models/model.onnx", tokenizer_path="data/tokenizer.json"):
        print("🤖 Khởi tạo Engine V4 [Pure Semantic Routing - Fixed Fallback]...")
        
        self.db_path = db_path
        self.intents_path = intents_path
        self.onnx_model_path = onnx_model_path
        self.tokenizer_path = tokenizer_path
        
        # 🏎️ TỐI ƯU HÓA ONNX RUNTIME DÀNH RIÊNG CHO RASPBERRY PI 4
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4  
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.ort_session = ort.InferenceSession(onnx_model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        self.l1_cache = {}       
        self.l2_cache = []       
        
        self.l2_threshold = 0.88       
        # 🎯 ĐIỀU CHỈNH CHIẾN LƯỢC: Nâng ngưỡng nghiêm ngặt lên 0.82.
        # Các câu hỏi kiến thức chung (đi mưa, chập điện) sẽ không bao giờ đạt độ tương đồng > 0.82 
        # với các mẫu cứng ("giá bao nhiêu", "thông số kỹ thuật"). Nhờ đó chúng sẽ rơi thẳng xuống Tầng 6 (LLM).
        self.global_intent_threshold = 0.82  
        self.max_l2_size = 300
        
        self.pc_llm_url = "http://192.168.1.91:11434/v1/chat"  
        self.reload_database()
        print("✅ Hệ thống Hybrid V4 (Thoát ly Keyword hoàn toàn) sẵn sàng!")

    def _normalize_for_cache(self, text):
        text = re.sub(r'[^\w\s]', ' ', text)  
        return " ".join(text.lower().strip().split())

    def reload_database(self):
        start_reload = time.time()
        print("🔄 [V4] Đang nạp cấu hình và tiến hành Vector hóa siêu ngữ nghĩa...")
        
        with open(self.db_path, "r", encoding="utf-8") as f:
            self.db = json.load(f)
        with open(self.intents_path, "r", encoding="utf-8") as f:
            self.intents_config = json.load(f)
            
        self.l1_cache.clear()
        self.l2_cache.clear()

        # 1. Vector hóa Ý định từ JSON (Dựa hoàn toàn vào mẫu câu Semantic, không dùng Keyword)
        self.intent_matrices = {}
        for intent_key, data in self.intents_config.get("business_rules", {}).items():
            samples = data.get("samples", [])
            if samples:
                vectors = [self._get_embedding(self._normalize_for_cache(phrase)) for phrase in samples]
                self.intent_matrices[intent_key] = np.vstack(vectors)
            
        # 2. Xây dựng từ điển thực thể phân cấp (Chỉ dùng Regex bóc tách danh từ riêng Hãng/Mẫu xe)
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

        # 🎯 3. THAY THẾ TOÀN BỘ MẢNG CÂU MẪU DÀI DÒNG BẰNG KHÁI NIỆM VĨ MÔ (MACRO CONCEPTS)
        # Giúp loại bỏ hoàn toàn các danh sách 'ev_anchor_phrases' và 'out_of_scope_phrases' thủ công trước đây.
        ev_macro_concepts = [
            "thông tin tra cứu bảng giá thông số chính sách bảo hành của các dòng xe máy điện",
            "hướng dẫn sử dụng sạc pin cứu hộ sửa chữa lỗi chập cháy ngập nước kỹ thuật xe điện"
        ]
        out_macro_concepts = [
            "công thức nấu ăn ẩm thực thời tiết hôm nay thời sự âm nhạc giải trí",
            "chơi game trực tuyến đá bóng thể thao toán học văn học khoa học tâm sự cuộc sống"
        ]
        
        self.ev_domain_matrix = np.vstack([self._get_embedding(self._normalize_for_cache(c)) for c in ev_macro_concepts])
        self.out_of_scope_matrix = np.vstack([self._get_embedding(self._normalize_for_cache(c)) for c in out_macro_concepts])

        print(f"⚡ Hoàn tất Pure Semantic Pipeline! Thời gian: {round(time.time() - start_reload, 3)} giây!")

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
        detected_brand = None
        for brand_key, brand_data in self.brands.items():
            for term in brand_data["search_terms"]:
                pattern = r'(?<![\w])' + re.escape(term) + r'(?![\w])'
                if re.search(pattern, clean_text):
                    detected_brand = brand_key
                    break
            if detected_brand: break
            
        if detected_brand:
            for model in self.brands[detected_brand]["models"]:
                for term in model["search_terms"]:
                    if term:
                        pattern = r'(?<![\w])' + re.escape(term) + r'(?![\w])'
                        if re.search(pattern, clean_text):
                            return (detected_brand, model["model_key"])
        else:
            for brand_key, brand_data in self.brands.items():
                for model in brand_data["models"]:
                    for term in model["search_terms"]:
                        if term:
                            pattern = r'(?<![\w])' + re.escape(term) + r'(?![\w])'
                            if re.search(pattern, clean_text):
                                return (brand_key, model["model_key"])
        return (detected_brand, None) if detected_brand else None

    def _is_ev_related(self, query_vector):
        in_domain_sim = np.max(np.dot(self.ev_domain_matrix, query_vector.T).flatten())
        out_domain_sim = np.max(np.dot(self.out_of_scope_matrix, query_vector.T).flatten())
        
        if out_domain_sim > in_domain_sim:
            return False
        return True

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
            print(f"⚠️ Thất bại khi kết nối tới LLM Server: {e}")
        return None

    def get_response(self, user_query):
        start_time = time.time()
        clean_query = self._normalize_for_cache(user_query)
        
        if not clean_query:
            return "Dạ em chào anh/chị, anh/chị cần hỗ trợ gì ạ?", round(time.time() - start_time, 4), "EMPTY_QUERY"
            
        # 🚀 TẦNG 1: L1 CACHE
        if clean_query in self.l1_cache:
            return self.l1_cache[clean_query], round(time.time() - start_time, 4), "L1_HIT"
            
        # 🚀 TẦNG 2: FAST FAQ TRIAGE (Chỉ giữ lại quét từ khóa cho Chào hỏi / Liệt kê hãng xe cơ bản)
        for intent_name, data in self.intents_config.get("faq", {}).items():
            pattern = r'(?<![\w])(' + '|'.join(re.escape(kw) for kw in data["keywords"]) + r')(?![\w])'
            if re.search(pattern, clean_query):
                final_reply = data["response_template"]
                self.l1_cache[clean_query] = final_reply
                return final_reply, round(time.time() - start_time, 4), "CACHE_MISS"

        current_entity = self._extract_entity(clean_query)
        query_vector = self._get_embedding(clean_query)

        # 🚀 TẦNG 3: SEMANTIC REJECT (Dựa hoàn toàn vào Zero-shot Macro Concept)
        if not self._is_ev_related(query_vector) and not current_entity:
            fallback_msg = "Dạ, hiện tại em chưa hiểu rõ ý của anh/chị. Anh/chị có thể hỏi cụ thể hơn về Giá cả, Thông số kỹ thuật hoặc Bảo hành của các dòng xe máy điện được không ạ?"
            self.l1_cache[clean_query] = fallback_msg
            return fallback_msg, round(time.time() - start_time, 4), "SEMANTIC_REJECT"

        # Định tuyến Ý định NLU
        # Định tuyến Ý định NLU cải tiến bằng Thuật toán Biên độ Ngữ nghĩa (Semantic Margin)
        current_intent = None
        intent_scores = {}
        for intent_key, matrix in self.intent_matrices.items():
            similarities = np.dot(matrix, query_vector.T).flatten()
            intent_scores[intent_key] = np.max(similarities)
            
        if intent_scores:
            # Sắp xếp các Intent theo điểm số giảm dần
            sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
            best_intent, max_intent_score = sorted_intents[0]
            second_intent, second_score = sorted_intents[1] if len(sorted_intents) > 1 else (None, 0)
            
            if max_intent_score >= self.global_intent_threshold:
                # Nếu KHÔNG CÓ thực thể xe cụ thể, bắt buộc phải kiểm tra Biên độ Ngữ nghĩa
                if not current_entity or not current_entity[1]:
                    semantic_margin = max_intent_score - second_score
                    # Nếu khoảng cách giữa 2 ý định quá hẹp (< 0.07), chứng tỏ mô hình đang phân vân do câu hỏi dạng kiến thức chung
                    if semantic_margin >= 0.07:
                        current_intent = best_intent
                    else:
                        print(f"🔀 [Semantic Margin] Phát hiện biên độ hẹp ({round(semantic_margin, 3)}). Hủy chặn Intent để chuyển tiếp xuống LLM.")
                        current_intent = None
                else:
                    current_intent = best_intent

        # 🛡️ MÀNG LỌC GIA CỐ: Khắc phục lỗi câu hỏi kiến thức chung / câu bẫy bị chặn cứng ở Tầng 5
        # Nếu nhận diện được Intent cứng nhưng câu hỏi THIẾU thực thể mẫu xe cụ thể (model_key là None),
        # hệ thống bắt buộc phải đối chiếu chéo với danh sách từ khoá (keywords) để tránh nhận diện sai do trùng lặp vector.
        if current_intent and (not current_entity or not current_entity[1]):
            intent_kws = self.intents_config.get("business_rules", {}).get(current_intent, {}).get("keywords", [])
            if intent_kws:
                # Kiểm tra xem có từ khoá cốt lõi nào của intent xuất hiện trong câu hỏi hay không
                has_keyword = any(re.search(r'(?<![\w])' + re.escape(kw) + r'(?![\w])', clean_query) for kw in intent_kws)
                if not has_keyword:
                    print(f"⚠️ [Guardrail] Phát hiện lệch hướng ngữ nghĩa! Huỷ bỏ intent '{current_intent}' giả lập để đẩy xuống tầng xử lý sâu hơn.")
                    current_intent = None  # Hoá giải Intent sai lệch, cho phép câu hỏi đi xuống các tầng dưới

        

        # 🚀 TẦNG 5: BUSINESS LOGIC & DB EXTRACTION (ĐÃ XOÁ SỔ HOÀN TOÀN CROSS-CHECK KEYWORD)
        final_reply = None
        cache_status = "CACHE_MISS"
        
        business_rules = self.intents_config.get("business_rules", {})
        if current_intent in business_rules:
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
                # Chỉ chặn cứng bằng JSON Response nếu điểm số đạt độ chắc chắn tuyệt đối (> 0.92)
                # Nếu điểm số chỉ ở mức trung bình (0.82 -> 0.92), cho phép trôi xuống LLM để xử lý linh hoạt hơn.
                if max_intent_score >= 0.92:
                    final_reply = business_rules[current_intent]["missing_entity_response"]
                else:
                    final_reply = None  # Nhường quyền cho LLM lập luận
        # 🚀 TẦNG 4: L2 CACHE
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

        # 🚀 TẦNG 6: PC FLASK LLM SERVER (Hệ thống sẽ rơi xuống đây mượt mà khi gặp câu hỏi khó)
        if not final_reply:
            print("🌐 Gửi yêu cầu phân tích chuyên sâu qua mạng LAN tới PC LLM...")
            llm_reply = self._call_pc_llm_via_lan(user_query)
            if llm_reply:
                if "OUT_OF_SCOPE_TRIGGERED" in llm_reply:
                    final_reply = "Dạ, hiện tại em chưa hiểu rõ ý của anh/chị. Anh/chị có thể hỏi cụ thể hơn về Giá cả, Thông số hoặc Bảo hành của xe điện được không ạ?"
                    cache_status = "LLM_GUARD_REJECT"
                else:
                    final_reply = llm_reply
                    cache_status = "PC_LLM_HIT"

        # 🚀 TẦNG 7: FALLBACK CUỐI CÙNG
        if not final_reply:
            final_reply = "Dạ, hiện tại em chưa hiểu rõ ý của anh/chị hoặc thông tin này nằm ngoài cơ sở dữ liệu. Anh/chị có thể hỏi cụ thể hơn về xe điện được không ạ?"

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