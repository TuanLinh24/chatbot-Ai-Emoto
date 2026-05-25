from flask import Flask, request, jsonify
from flasgger import Swagger
from logic_engine import ChatbotEngine_V4

# 1. Khởi tạo Flask App và Swagger UI
app = Flask(__name__)
app.config['SWAGGER'] = {
    'title': 'API Chatbot Xe Máy Điện Cải Tiến Cache L1/L2',
    'uiversion': 3
}
swagger = Swagger(app)

# 2. Khởi tạo Backend Engine
print("Khởi động Server với Flask và Swagger...")
engine = ChatbotEngine_V4(
    db_path="database_v2.json",
    intents_path="data/intents.json"
    # Lưu ý: Đảm bảo onnx_model_path và tokenizer_path mặc định trong logic_engine2.py 
    # khớp với đường dẫn thực tế của bạn, hoặc bạn có thể truyền thẳng vào đây.
)

# 3. API Endpoint Chatbot gốc
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
              example: "xe Yadea Phantom Max giá bao nhiêu"
            user_id:
              type: string
              description: "ID của người dùng (tùy chọn)"
              example: "khachhang_01"
    responses:
      200:
        description: Trả về câu trả lời thành công kèm trạng thái bộ nhớ đệm chi tiết
        schema:
          type: object
          properties:
            reply:
              type: string
              example: "Dạ, mẫu xe Yadea Phantom Max có giá là 29.900.000 VNĐ."
            status:
              type: string
              example: "success"
            metrics:
              type: object
              properties:
                response_time_seconds:
                  type: number
                  format: float
                  example: 0.025
                cache_status:
                  type: string
                  description: "Trạng thái bộ nhớ đệm: L1_HIT, L2_HIT (kèm % đồng nghĩa), hoặc CACHE_MISS"
                  example: "L2_HIT (94.2%)"
      400:
        description: Thiếu tham số message
      500:
        description: Lỗi máy chủ (Server Error)
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "status": "error", 
                "reply": "Thiếu trường 'message' trong dữ liệu gửi lên."
            }), 400
            
        user_message = data['message']
        
        # Nhận diện trạng thái cache dạng chuỗi từ logic_engine mới
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

# 4. API Endpoint Reload mới để đồng bộ dữ liệu từ giao diện UI
@app.route('/api/reload', methods=['POST'])
def reload_endpoint():
    """
    Kích hoạt Hot-Reload đồng bộ tri thức từ UI CMS và làm sạch Cache hoàn toàn
    ---
    tags:
      - Admin System Tools
    responses:
      200:
        description: Đồng bộ dữ liệu và dọn sạch Cache thành công!
        schema:
          type: object
          properties:
            status:
              type: string
              example: "success"
            message:
              type: string
              example: "Hệ thống AI Robot đã đồng bộ và dọn sạch bộ nhớ đệm L1/L2 thành công!"
      500:
        description: Lỗi nạp cấu hình cơ sở dữ liệu (JSON lỗi hoặc thiếu file)
    """
    try:
        engine.reload_database()
        return jsonify({
            "status": "success",
            "message": "Hệ thống AI Robot đã đồng bộ và dọn sạch bộ nhớ đệm L1/L2 thành công!"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Lỗi nạp cấu hình dữ liệu: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)