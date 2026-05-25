import requests
import time
import json

# Cấu hình Endpoint của Chatbot Server (server2.py)
API_URL = "http://127.0.0.1:8000/api/chat"
RELOAD_URL = "http://127.0.0.1:8000/api/reload"

# Định nghĩa các ca kiểm thử (Test Cases) đại diện cho toàn bộ tính năng
TEST_CASES = [
    # 1. Kiểm thử Nhóm Chào hỏi / FAQ Hệ thống
    {
        "id": 1,
        "category": "FAQ / Intent Gốc",
        "query": "xin chào em",
        "expected_status": ["PC_LLM_HIT", "L1_HIT", "CACHE_MISS"], 
        "description": "Kiểm tra khả năng định tuyến câu chào hỏi cơ bản"
    },
    
    # 2. Kiểm thử Truy vấn Cơ sở dữ liệu JSON (Cơ chế Extraction & Quy tắc Rule)
    {
        "id": 2,
        "category": "JSON DB Query (Gia Ca)",
        "query": "Giá xe VinFast Pansy V niêm yết bao nhiêu tiền vậy",
        "expected_status": ["CACHE_MISS"],
        "description": "Truy xuất chính xác trường dữ liệu giá của thực thể VinFast Pansy V (Lần đầu - Miss)"
    },
    {
        "id": 3,
        "category": "JSON DB Query (Thong So)",
        "query": "Cho tôi xin thông số kỹ thuật của chiếc VinFast Beta",
        "expected_status": ["CACHE_MISS"],
        "description": "Truy xuất thuộc tính kỹ thuật của thực thể VinFast Beta"
    },
    {
        "id": 4,
        "category": "JSON DB Query (Bao Hanh)",
        "query": "Chính sách bảo hành xe Sym Theon Lite thế nào",
        "expected_status": ["CACHE_MISS"],
        "description": "Truy xuất thuộc tính bảo hành của thực thể Sym Theon Lite"
    },

    # 3. Kiểm thử Tầng Bộ nhớ đệm L1 Cache (Trùng khớp 100%)
    {
        "id": 5,
        "category": "L1 Cache Hit",
        "query": "Giá xe VinFast Pansy V niêm yết bao nhiêu tiền vậy",
        "expected_status": ["L1_HIT"],
        "description": "Yêu cầu lặp lại y hệt câu số 2 để kiểm tra phản hồi tức thì từ bộ nhớ L1 (< 2ms)"
    },

    # 4. Kiểm thử Tầng Bộ nhớ đệm L2 Cache (Khác câu từ - Trùng ngữ nghĩa Vector)
    {
        "id": 6,
        "category": "L2 Cache Hit",
        "query": "Cho anh hỏi con Pansy V của VinFast giá bao nhiêu thế",
        "expected_status": ["L2_HIT"],
        "description": "Đổi cấu trúc câu hỏi của câu số 2 để kiểm tra độ tương đồng không gian Vector L2"
    },

    # 5. Kiểm thử Quy tắc điền thông tin (Rule Missing Entity Prompting)
    {
        "id": 7,
        "category": "Rule Constraint",
        "query": "Tôi muốn xem bảng giá bán",
        "expected_status": ["CACHE_MISS"],
        "description": "Có Intent giá cả nhưng thiếu Tên xe -> Kích hoạt Rule nhắc nhở gọi tên hãng"
    },

    # 6. Kiểm thử Màng lọc Ngữ nghĩa chặn đứng các câu ngoài lề (Out-of-Scope Rejection)
    {
        "id": 8,
        "category": "Out-of-Scope Block",
        "query": "xe máy điện phở bò nấu thế nào cho ngon",
        "expected_status": ["SEMANTIC_REJECT", "LLM_GUARD_REJECT"],
        "description": "Bẫy từ khóa trộn lẫn 'xe máy điện' và 'nấu phở' -> Phải bị màng lọc đánh chặn cục bộ"
    },
    {
        "id": 9,
        "category": "Out-of-Scope Block",
        "query": "Thời tiết ngày mai ở Sài Gòn có mưa đá bóng không",
        "expected_status": ["SEMANTIC_REJECT", "LLM_GUARD_REJECT"],
        "description": "Câu hỏi rác hoàn toàn lạc đề về thời tiết, thể thao -> Đánh chặn lập tức"
    },

    # 7. Kiểm thử Định tuyến LLM Chuyên sâu (Đúng chủ đề EV nhưng không có trong JSON)
    {
        "id": 10,
        "category": "LAN PC LLM Fallback",
        "query": "Xe điện đi trời mưa ngập nước sâu có dễ bị chập cháy nổ pin LFP không em",
        "expected_status": ["PC_LLM_HIT"],
        "description": "Kiến thức chuyên sâu ngoài DB cứng -> Chuyển vùng tính toán mạng LAN sang PC LLM"
    }
]

def run_performance_test():
    print("===============================================================================")
    print("🚨 HỆ THỐNG KIỂM THỬ TOÀN DIỆN CHATBOT HYBRID LAN V4 - AUTOMATION TESTING TOOL")
    print("===============================================================================")
    
    # Kích hoạt Hot-Reload để làm sạch bộ nhớ đệm Cache trước khi Test
    print("\n🔄 Đang gửi tín hiệu Hot-Reload dọn sạch Cache hệ thống về trạng thái Nguyên Bản...")
    try:
        reload_res = requests.post(RELOAD_URL, timeout=5)
        if reload_res.status_code == 200:
            print("✅ Làm sạch L1/L2 Cache thành công! Bắt đầu chu trình Test.\n")
        else:
            print("⚠️ Cảnh báo: Không thể gọi lệnh Reload dữ liệu. Kết quả test cache có thể bị ảnh hưởng.\n")
    except Exception as e:
        print(f"❌ Không kết nối được API Server tại {RELOAD_URL}. Vui lòng bật 'python server2.py' trước!\n")
        return

    test_results = []
    total_time = 0
    passed_cases = 0

    for idx, tc in enumerate(TEST_CASES):
        print(f"👉 [Test Case {tc['id']}/{len(TEST_CASES)}] [{tc['category']}]")
        print(f"   💬 Câu hỏi: \"{tc['query']}\"")
        print(f"   ℹ️  Mục tiêu: {tc['description']}")
        
        start_local_time = time.time()
        try:
            response = requests.post(API_URL, json={"message": tc["query"]}, timeout=15)
            local_duration = time.time() - start_local_time
            
            if response.status_code == 200:
                res_json = response.json()
                reply = res_json.get("reply", "")
                metrics = res_json.get("metrics", {})
                
                # Trích xuất dữ liệu trả về từ Server
                server_duration = metrics.get("response_time_seconds", local_duration)
                cache_status = metrics.get("cache_status", "UNKNOWN")
                
                # Xác định trạng thái Đạt/Không đạt (Pass/Fail)
                is_passed = any(status in cache_status for status in tc["expected_status"])
                status_color = "🟢 PASS" if is_passed else "🔴 FAIL"
                
                if is_passed:
                    passed_cases += 1
                total_time += server_duration
                
                print(f"   ⏱️  Thời gian phản hồi: {round(server_duration, 4)} giây")
                print(f"   🛡️  Tầng xử lý thực tế: {cache_status} (Kỳ vọng: {tc['expected_status']})")
                print(f"   🤖 Chatbot phản hồi: \"{reply[:90]}...\"")
                print(f"   📊 Kết quả: {status_color}\n")
                
                test_results.append({
                    "id": tc["id"],
                    "category": tc["category"],
                    "query": tc["query"],
                    "cache_status": cache_status,
                    "duration": round(server_duration, 4),
                    "result": "PASS" if is_passed else "FAIL"
                })
            else:
                print(f"   🔴 Lỗi Server: HTTP Status Code {response.status_code}\n")
                test_results.append({
                    "id": tc["id"], "category": tc["category"], "query": tc["query"],
                    "cache_status": f"HTTP_{response.status_code}", "duration": 0.0, "result": "FAIL"
                })
                
        except requests.exceptions.Timeout:
            print("   🔴 Kết quả: FAIL (Mạng nội bộ LAN LLM bị quá tải/Timeout hiển thị)\n")
            test_results.append({
                "id": tc["id"], "category": tc["category"], "query": tc["query"],
                "cache_status": "TIMEOUT", "duration": 15.0, "result": "FAIL"
            })
        except Exception as e:
            print(f"   🔴 Lỗi kết nối vật lý: {str(e)}\n")
            test_results.append({
                "id": tc["id"], "category": tc["category"], "query": tc["query"],
                "cache_status": "ERROR", "duration": 0.0, "result": "FAIL"
            })

    # =========================================================================
    # BẢNG TỔNG HỢP KẾT QUẢ ĐẦU RA (Ứng dụng định dạng chuỗi thuần không lỗi font)
    # =========================================================================
    print("\n" + "="*105)
    print(f"📊 BẢNG TỔNG HỢP ĐÁNH GIÁ NĂNG LỰC VẬN HÀNH CHATBOT (KPI REPORT)")
    print("="*105)
    header_fmt = "| %-3s | %-22s | %-45s | %-16s | %-7s | %-5s |"
    row_fmt = "| %-3d | %-22s | %-45s | %-16s | %-7.4f | %-5s |"
    
    print(header_fmt % ("STT", "Hạng mục kiểm thử", "Câu hỏi đầu vào", "Tầng định tuyến", "Time(s)", "Result"))
    print("-" * 105)
    
    for row in test_results:
        # Cắt ngắn câu hỏi nếu quá dài để giữ form bảng ngay ngắn
        short_query = row["query"] if len(row["query"]) <= 42 else row["query"][:40] + "..."
        print(row_fmt % (row["id"], row["category"], short_query, row["cache_status"], row["duration"], row["result"]))
        
    print("="*105)

    # Vùng Chỉ số đo lường hiệu năng cốt lõi (Core Metrics Evaluation)
    accuracy_rate = (passed_cases / len(TEST_CASES)) * 100
    avg_latency = total_time / len(TEST_CASES) if len(TEST_CASES) > 0 else 0
    
    print(f"📈 TỶ LỆ CHÍNH XÁC ĐỊNH TUYẾN CHỨC NĂNG : {accuracy_rate:.1f}% ({passed_cases}/{len(TEST_CASES)} Ca thành công)")
    print(f"⏱️ TỐC ĐỘ PHẢN HỒI TRUNG BÌNH HỆ THỐNG   : {avg_latency:.4f} giây / câu hỏi")
    print("="*105)

    # =========================================================================
    # HỆ THỐNG ĐÁNH GIÁ CHUYÊN SÂU TỪ AI (EVALUATION SUMMARY)
    # =========================================================================
    print("\n🧐 BÁO CÁO PHÂN TÍCH VÀ ĐÁNH GIÁ CHẤT LƯỢNG CHATBOT:")
    
    # 1. Đánh giá tính năng Chặn đứng các câu hỏi ngoài lề / Bẫy ngữ nghĩa
    print("  🔹 [Màng Lọc Chặn Bẫy & Lạc Đề]: ", end="")
    if test_results[7]["result"] == "PASS" and test_results[8]["result"] == "PASS":
        print("HOÀN HẢO! Chatbot nhận diện ngữ nghĩa xuất sắc. Các câu hỏi bẫy từ khóa ('xe máy điện phở bò nấu thế nào') đã bị chặn đứng ngay lập tức tại Edge hoặc LLM Server, không sinh dữ liệu rác.")
    else:
        print("CẦN CẢI THIỆN! Bộ lọc ma trận tương phản chưa đủ mạnh hoặc ngưỡng Threshold đặt quá thấp khiến câu hỏi rác lọt lưới.")

    # 2. Đánh giá tính năng Lưu trữ đệm (Cache L1 & L2)
    print("  🔹 [Cơ Chế Tối Ưu Tốc Độ Cache L1/L2]: ", end="")
    l1_speed = test_results[4]["duration"]
    l2_speed = test_results[5]["duration"]
    if test_results[4]["result"] == "PASS" and test_results[5]["result"] == "PASS" and l1_speed < 0.05:
        print(f"XUẤT SẮC! Tầng L1 Cache phản hồi siêu tốc ({l1_speed}s). Tầng L2 Cache nhận diện đúng biến thể câu hỏi bằng Embedding đạt tỷ lệ khớp chuẩn.")
    else:
        print("CÓ LỖI: Thời gian xử lý Cache hoặc cơ chế định vị Vector L2 gặp trục trặc khiến thuật toán chạy lại từ đầu.")

    # 3. Đánh giá tính năng Kết hợp LAN LLM Hybrid
    print("  🔹 [Liên Kết Mạng Nội Bộ LAN LLM Fallback]: ", end="")
    if test_results[9]["cache_status"] == "PC_LLM_HIT":
        print(f"ỔN ĐỊNH. Khi gặp câu hỏi nằm ngoài JSON nhưng đúng phạm vi xe điện, hệ thống đã ủy quyền thành công cho GPU PC xử lý trong {test_results[9]['duration']}s.")
    else:
        print("THẤT BẠI. Kết nối LAN LLM Flask bị lỗi hoặc cấu trúc rẽ nhánh rớt xuống tầng Fallback mặc định.")

    # Kết luận KPI Chung
    print("\n🏆 KẾT LUẬN CHUNG: ", end="")
    if accuracy_rate >= 90 and avg_latency < 1.5:
        print("ĐẠT CHUẨN SẢN PHẨM CAO CẤP (Production-Ready). Toàn bộ luồng hoạt động phân tầng hoạt động chính xác theo đúng thiết kế kiến trúc máy trạm Pi 4 & Máy chủ GPU.")
    elif accuracy_rate >= 70:
        print("ĐẠT CHUẨN THỬ NGHIỆM (Beta Version). Chatbot vận hành ổn định nhưng cần tối ưu hóa thêm tốc độ phần cứng hoặc tinh chỉnh lại các tham số Threshold.")
    else:
        print("CHƯA ĐẠT CHUẨN. Cần rà soát lại file 'logic_engine2.py' vì cấu trúc định tuyến đang sinh lỗi logic diện rộng.")
    print("="*105 + "\n")

if __name__ == "__main__":
    run_performance_test()