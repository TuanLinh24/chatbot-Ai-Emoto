from flask import Flask, request, jsonify
from llama_cpp import Llama
import time

app = Flask(__name__)

MODEL_PATH = "models/qwen2.5-3b-instruct-q4_k_m.gguf" 

print("🔄 Đang nạp mô hình Qwen 2.5 3B vào VRAM GPU...")
llm = Llama(
    model_path=MODEL_PATH, 
    n_ctx=1024,          
    n_gpu_layers=-1,     # Đẩy 100% lên GPU GTX 1650
    verbose=False        
)
print("✅ Hệ thống LLM Server sẵn sàng lắng nghe mạng LAN.")

@app.route('/v1/chat', methods=['POST'])
def process_chat_llm():
    start_time = time.time()
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"status": "error", "reply": "Thiếu tham số 'message'"}), 400
            
        user_message = data['message']
        
        # 🎯 THIẾT LẬP CHẶN BẪY NGỮ NGHĨA BẰNG SYSTEM PROMPT (CHATML CHUẨN QWEN)
        full_prompt = (
            "<|im_start|>system\n"
            "Bạn là trợ lý ảo chuyên gia tư vấn xe máy điện lịch sự, chuyên nghiệp.\n"
            "QUY TẮC BẮT BUỘC:\n"
            "1. Chỉ trả lời các kiến thức liên quan trực tiếp đến xe điện, pin, trạm sạc hoặc kỹ thuật xe máy điện.\n"
            "2. Trả lời ngắn gọn, trực diện vấn đề trong tối đa 2 đến 3 câu.\n"
            "3. Nếu câu hỏi hoàn toàn KHÔNG liên quan đến xe điện, lạc đề, hoặc là câu hỏi bẫy vô nghĩa (Ví dụ: 'xe điện nấu phở thế nào', 'hôm nay ăn gì', 'làm thơ đi'...), bạn PHẢI trả lời chính xác cụm từ: OUT_OF_SCOPE_TRIGGERED\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        output = llm(
            full_prompt,
            max_tokens=150,      
            temperature=0.0,     # Hạ thấp tối đa sự sáng tạo để kiểm soát cấu trúc đầu ra nghiêm ngặt
            top_p=0.85,
            stop=["<|im_end|>", "<|endoftext|>", "</s>"] 
        )
        
        reply_content = output['choices'][0]['text'].strip()
        process_duration = time.time() - start_time
        
        return jsonify({
            "status": "success",
            "reply": reply_content,
            "metrics": {
                "pc_inference_time_seconds": round(process_duration, 3)
            }
        }), 200
        
    except Exception as e:
        print(f"❌ Lỗi xử lý LLM: {str(e)}")
        return jsonify({
            "status": "error", 
            "reply": "Dạ, hệ thống xử lý thông tin chuyên sâu đang bận, anh/chị thử lại sau nhé!"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=11434, debug=False)