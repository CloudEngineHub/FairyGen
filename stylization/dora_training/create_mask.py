import os
import io
from PIL import Image
from rembg import new_session, remove
import numpy as np

input_image_path = '/Users/zhengjiayi/Downloads/FairyGen/stylization/dora_training/data/train/green_boy/texture.png'  
output_mask_path = '/Users/zhengjiayi/Downloads/FairyGen/stylization/dora_training/data/train/green_boy/mask.png'                         
model_name = "isnet-anime"                                    

def process_single_image(input_path, output_path, model_name):
    session = new_session(model_name)
    
    try:
        if not os.path.exists(input_path):
            print(f"错误: 找不到文件 {input_path}")
            return

        with open(input_path, 'rb') as i:
            input_data = i.read()
            
        mask_data = remove(input_data, session=session, only_mask=True)
        
        mask_image = Image.open(io.BytesIO(mask_data)).convert("L")
        mask_array = np.array(mask_image, dtype=np.uint8)
        
        mask_binary = (mask_array > 127).astype(np.uint8) * 255
        final_mask = Image.fromarray(mask_binary, mode="L")
        
        final_mask.save(output_path)
        print(f"成功！Mask 已保存至: {output_path}")

    except Exception as e:
        print(f"处理过程中出错: {e}")

if __name__ == "__main__":
    process_single_image(input_image_path, output_mask_path, model_name)
