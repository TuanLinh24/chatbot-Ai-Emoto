#!/usr/bin/env python3
"""
Kiểm tra tham số model GGUF để đánh giá khả năng chạy trên Raspberry Pi 4
"""
from pathlib import Path
import struct
import os

def check_model_params():
    model_dir = Path("models")
    
    # Tìm file GGUF
    gguf_files = list(model_dir.glob("*.gguf"))
    
    if not gguf_files:
        print("❌ Không tìm thấy file GGUF trong thư mục models/")
        return
    
    model_file = gguf_files[0]
    print(f"📦 Model: {model_file.name}")
    
    # Kích thước file
    file_size_mb = model_file.stat().st_size / (1024 * 1024)
    print(f"📏 Kích thước file: {file_size_mb:.2f} MB")
    
    # Ước tính bộ nhớ cần thiết (file size * 1.5 cho loading + context)
    estimated_memory_mb = file_size_mb * 2
    print(f"💾 Bộ nhớ cần thiết (ước tính): {estimated_memory_mb:.2f} MB")
    
    # Đọc header GGUF để lấy metadata
    try:
        with open(model_file, 'rb') as f:
            # GGUF magic number
            magic = f.read(4)
            if magic != b'GGUF':
                print(f"❌ File không phải GGUF format, magic: {magic}")
                return
            
            # Version
            version = struct.unpack('<I', f.read(4))[0]
            print(f"📋 GGUF Version: {version}")
            
            # Tensor count và KV pairs count
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            kv_count = struct.unpack('<Q', f.read(8))[0]
            print(f"📊 Tensor count: {tensor_count}")
            print(f"📊 KV pairs: {kv_count}")
            
            # Đọc KV pairs để tìm các thông số quan trọng
            kv_params = {}
            for _ in range(kv_count):
                # Key length
                key_len = struct.unpack('<I', f.read(4))[0]
                key = f.read(key_len).decode('utf-8', errors='ignore').rstrip('\0')
                
                # Value type
                value_type = struct.unpack('<I', f.read(4))[0]
                
                # Read value based on type
                value = None
                if value_type == 0:  # uint32
                    value = struct.unpack('<I', f.read(4))[0]
                elif value_type == 1:  # int32
                    value = struct.unpack('<i', f.read(4))[0]
                elif value_type == 2:  # uint64
                    value = struct.unpack('<Q', f.read(8))[0]
                elif value_type == 3:  # int64
                    value = struct.unpack('<q', f.read(8))[0]
                elif value_type == 4:  # float32
                    value = struct.unpack('<f', f.read(4))[0]
                elif value_type == 5:  # bool
                    value = f.read(1)[0] != 0
                elif value_type == 6:  # string
                    str_len = struct.unpack('<I', f.read(4))[0]
                    value = f.read(str_len).decode('utf-8', errors='ignore')
                
                kv_params[key] = value
            
            # In các tham số quan trọng
            print("\n🔧 Tham số model:")
            important_keys = [
                'llama.context_length',
                'llama.embedding_length',
                'llama.block_count',
                'llama.attention.head_count',
                'llama.attention.head_count_kv',
                'llama.feed_forward_length',
                'general.file_type',
                'general.name',
                'model_params'
            ]
            
            for key in important_keys:
                if key in kv_params:
                    print(f"  {key}: {kv_params[key]}")
            
            # Thông tin quantization
            file_type = kv_params.get('general.file_type', 'unknown')
            quantization_types = {
                0: 'F32 (float32)',
                1: 'F16 (float16)',
                2: 'Q4_0',
                3: 'Q4_1',
                6: 'Q5_0',
                7: 'Q5_1',
                8: 'Q8_0',
                12: 'Q2_K',
                13: 'Q3_K_S',
                14: 'Q3_K_M',
                15: 'Q4_K_M',
                16: 'Q5_K_S',
                17: 'Q5_K_M',
                18: 'Q6_K',
            }
            quant_name = quantization_types.get(file_type, f'Type {file_type}')
            print(f"\n⚙️  Loại lượng tử hóa: {quant_name}")
            
    except Exception as e:
        print(f"❌ Lỗi khi đọc file: {e}")
        return
    
    # Đánh giá cho Raspberry Pi 4
    print("\n📱 Đánh giá cho Raspberry Pi 4:")
    print("=" * 50)
    
    # Raspberry Pi 4 thường có 2-8GB RAM
    rpi4_ram_options = [2, 4, 8]
    
    for ram_gb in rpi4_ram_options:
        ram_mb = ram_gb * 1024
        
        # Bỏ dành ~300MB cho hệ điều hành
        available_mb = ram_mb - 300
        
        if estimated_memory_mb < available_mb:
            print(f"✅ RPi 4 {ram_gb}GB RAM: CÓ THỂ CHẠY")
            print(f"   Model cần: {estimated_memory_mb:.0f}MB, còn lại: {available_mb - estimated_memory_mb:.0f}MB")
        else:
            print(f"⚠️  RPi 4 {ram_gb}GB RAM: KHÓ KHĂN (thiếu {estimated_memory_mb - available_mb:.0f}MB)")
    
    print("\n💡 Ghi chú:")
    print("  • Model nhẹ (Q4_K_M): phù hợp cho CPU yếu")
    print("  • Tốc độ inference sẽ chậm (~10-30 tokens/giây trên CPU)")
    print("  • Nên chạy với batch_size=1 để tiết kiệm bộ nhớ")
    print("  • Xem xét overclocking RPi 4 hoặc thêm swap nếu cần")

if __name__ == "__main__":
    check_model_params()
