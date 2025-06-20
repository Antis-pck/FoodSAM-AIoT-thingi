import os
import streamlit as st
from pathlib import Path
import subprocess
import time

def setup_image_directory():
    """Setup the required directory structure"""
    img_dir = os.path.join('dataset', 'FoodSeg103', 'Images', 'img_dir', 'test')
    os.makedirs(img_dir, exist_ok=True)
    return img_dir

def process_uploaded_image(uploaded_file, target_dir):
    if uploaded_file is None:
        return None
    
    file_extension = Path(uploaded_file.name).suffix
    target_path = os.path.join(target_dir, f"test{file_extension}")
    
    # Clean directory
    for existing_file in os.listdir(target_dir):
        os.remove(os.path.join(target_dir, existing_file))
    
    # Save new file
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return target_path
def run_semantic_script():
    """Run the semantic.py script using subprocess with all required arguments"""
    try:
        cmd = [
            'python', 
            'FoodSAM/semantic.py',
            '--device', 'cuda',
            '--data_root', 'dataset',
            '--img_dir', 'FoodSeg103/Images/img_dir/test',
            '--output', 'Output/Semantic_Results',
            '--SAM_checkpoint', 'ckpts/sam_vit_h_4b8939.pth',
            '--semantic_config', 'configs/SETR_MLA_768x768_80k_base.py',
            '--semantic_checkpoint', 'ckpts/SETR_MLA/iter_80000.pth',
            '--model-type', 'vit_h',
            '--category_txt', 'FoodSAM/FoodSAM_tools/category_id_files/foodseg103_category_id.txt',
            '--color_list_path', 'FoodSAM/FoodSAM_tools/color_list.npy',
            '--num_class', '104',
            '--area_thr', '0.0', 
            '--ratio_thr', '0.5',
            '--top_k', '80'  
        ]
        
        if True:  
            sam_settings = [
                '--points-per-side', '32',
                '--points-per-batch', '64',
                '--pred-iou-thresh', '0.88',
                '--stability-score-thresh', '0.95',
                '--stability-score-offset', '1.0',
                '--box-nms-thresh', '0.7',
                '--crop-n-layers', '1',
                '--crop-nms-thresh', '0.7',
                '--crop-overlap-ratio', '0.5',
                '--crop-n-points-downscale-factor', '2',
                '--min-mask-region-area', '100'
            ]
            cmd.extend(sam_settings)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.abspath(os.path.dirname(_file_))
        )
        
        log_output = ""
        if result.stdout:
            log_output += result.stdout
        if result.stderr:
            log_output += "\n" + result.stderr

        return True, log_output
        
    except subprocess.CalledProcessError as e:
        log_output = f"Error running semantic.py: {str(e)}"
        if e.stdout:
            log_output += "\n" + e.stdout
        if e.stderr:
            log_output += "\n" + e.stderr
        return False, log_output
    
def read_detected_categories():
    """Read and return actually detected categories from the semantic masks file"""
    try:
        mask_label_file = os.path.join('Output', 'Semantic_Results', 'test', 'sam_mask_label', 'semantic_masks_category.txt')
        if os.path.exists(mask_label_file):
            detected = set() 
            with open(mask_label_file, 'r', encoding='utf-8') as f:
                next(f) 
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:  
                        category_name = parts[2].strip()
                        confidence = float(parts[3])  
                        area = float(parts[4])
                        
                        if (category_name.lower() != 'background' and 
                            confidence > 0.5 and  
                            area > 0.001):      
                            detected.add(category_name)

            return sorted(list(detected))
    except Exception as e:
        st.error(f"Error reading categories: {str(e)}")
    return []

def main():
    st.title("Food Ingredient Classifier")
    
    img_dir = setup_image_directory()
    
    uploaded_file = st.file_uploader(
        "Upload a food image - JPG format preferred", 
        type=["jpg"]
    )

    if uploaded_file is not None:
        try:
            image_path = process_uploaded_image(uploaded_file, img_dir)
            
            if image_path:
                st.write("### Original Image")
                st.image(image_path, caption="Uploaded Image")

                if st.button("Process Image"):
                    with st.spinner("Processing image with FoodSAM..."):
                        success, log_output = run_semantic_script()
                        if success:
                            time.sleep(2) 
                            
                            # Display results
                            output_dir = 'Output/Semantic_Results'
                            if os.path.exists(output_dir):
                                result_img = os.path.join(output_dir, 'enhance_vis.png')
                                if os.path.exists(result_img):
                                    st.write("### Detection Results")
                                    st.image(result_img, caption="Processed Image")
                                
                                # Show detected categories
                                categories = read_detected_categories()
                                if categories:
                                    st.write("### Detected Ingredients:")
                                    for category in categories:
                                        st.write(f"- {category}")
                                else:
                                    st.warning("No ingredients detected.")
                            else:
                                st.error("Output directory not found.")
                        else:
                            st.error("Failed to process image.")

                        # Log output in an expander (collapsible tab)
                        with st.expander("Output", expanded=False):
                            st.text(log_output)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if _name_ == "_main_":
    main()