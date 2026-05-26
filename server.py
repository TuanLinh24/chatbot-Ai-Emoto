from flask import Flask, request, jsonify
from flasgger import Swagger
from logic_engine import ChatbotEngine_V4

# 1. Khởi tạo Flask App và Swagger UI
app = Flask(__name__)
app.config['SWAGGER'] = {
    'title': 'API Chatbot Xe Máy Điện Cải Tiến LAN Hybrid & Cache Ma Trận',
    'uiversion': 3
}
swagger = Swagger(app)

# 2. Khởi tạo Backend Engine
print("Khởi động Server kết hợp Edge-Inference và LAN LLM Routing...")
engine = ChatbotEngine_V4(
    db_path="database_v2.json",
    intents_path="data/intents.json"
)

# 3. API Endpoint Chatbot
@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """
    Gửi tin nhắn đến Chatbot và nhận câu trả lời
    ---
    tags:
      - Chatbot API
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            message:
              type: string
              description: "Câu hỏi của khách hàng"
              example: "xe máy điện đi ngập nước có sao không"
    responses:
      200:
        description: Trả về câu trả lời từ CSDL nội bộ hoặc từ Server LLM thông qua mạng nội bộ
        schema:
          type: object
          properties:
            reply:
              type: string
            status:
              type: string
            metrics:
              type: object
              properties:
                response_time_seconds:
                  type: number
                cache_status:
                  type: string
                  description: "Trạng thái xử lý: L1_HIT, L2_HIT, PC_LLM_HIT hoặc CACHE_MISS"
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "status": "error", 
                "reply": "Thiếu trường 'message' trong dữ liệu gửi lên."
            }), 400
            
        user_message = data['message']
        
        # Nhận câu trả lời và trạng thái định tuyến từ Engine mới
        bot_reply, process_time, cache_status = engine.get_response(user_message)
        
        return jsonify({
            "reply": bot_reply,
            "status": "success",
            "metrics": {
                "response_time_seconds": process_time,
                "cache_status": cache_status
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "reply": f"Lỗi server tại endpoint chat: {str(e)}"
        }), 500

# 4. API Endpoint Hot-Reload đồng bộ tri thức từ CMS
@app.route('/api/reload', methods=['POST'])
def reload_endpoint():
    """
    Kích hoạt Hot-Reload đồng bộ tri thức từ UI CMS và làm sạch Cache hoàn toàn
    ---
    tags:
      - Admin System Tools
    responses:
      200:
        description: Đồng bộ dữ liệu cục bộ và xây dựng lại không gian ma trận thành công!
    """
    try:
        engine.reload_database()
        return jsonify({
            "status": "success",
            "message": "Hệ thống AI Robot đã đồng bộ dữ liệu mạng LAN và dọn sạch bộ nhớ đệm L1/L2 thành công!"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Lỗi nạp cấu hình dữ liệu: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Chạy trên tất cả interfaces mạng nội bộ tại port 8000
    app.run(host='0.0.0.0', port=8000, debug=False)