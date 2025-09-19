
from transformers import AutoModel, GenerationConfig
from typing import Union
from prompts_chirag import prompts
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
import os
import re
from torchvision import transforms
import time
from PIL import Image
from utils import extract_cropped_human_frames, extract_face_human_frames
import glob
import ast

def check_tags(output: str) -> bool:
    pattern = r'<think>\s*.*?\s*</think>\s*<answer>\s*.*?\s*</answer>'
    match = re.search(pattern, output)
    return bool(match)


#stripping and splitting but not handling inconsistent use of ', "

# def parse_to_dict(output: str) -> dict:
#     output = output.strip().strip('{}') #stripping of any spaces front and back | extracting content in the curly braces
#     items = output.split(',')
#     result = {}
#     for item in items:
#         if ':' in item:
#             key, value = item.split(':', 1) #argument 1 specifies the maximum number of splits to perform
#             result[key.strip()] = value.strip()    
#     return result

def parse_to_dict(output: str) -> dict:
    output = output.strip().strip('{}') #stripping of any spaces front and back | extracting content in the curly braces
    items = output.split(',')
    result = {}
    for item in items:
        if ':' in item:
            key, value = item.split(':', 1)
            # remove extra spaces and surrounding quotes
            key = key.strip().strip('"').strip("'")
            value = value.strip().strip('"').strip("'")
            result[key] = value
    return result

def clean_output(report: dict, classes: list)-> dict:
    cleaned_report = {}
    for key, value in report.items():
        try:
            match = next((cls for cls in classes if cls.lower() in value.lower()))
            cleaned_report[key] = match if match else value
        except:
            cleaned_report[key] = value
    return cleaned_report


#works on a directory of frames, directory of videos, list of PIL Images

"""def triage_trauma(video: Union[list, str], model):

    # Important aspects of generation - bf16, length, temperature

    # gathering the prompts

    if isinstance(video, list): #handling list of PIL Images
        if len(video)%8 !=0:
            pad_len = 8 - (len(video) % 8)
            video.extend([frames[-1].copy() for _ in range(pad_len)])

        model.config.num_video_frames, model.config.fps = len(video), 0
        model.config.llm_cfg['use_bfloat16'] = True

        return forward_pass(frames, model)
    

    elif os.path.isdir(video): #handling dir of images
        frames_paths = sorted([
                os.path.join(video, f) 
                for f in os.listdir(video) 
                if os.path.splitext(f)[1].lower() in [".jpg", ".png", ".jpeg"]
            ])
        frames = [Image.open(f).convert("RGB") for f in frames_paths]

        if len(frames)%8 !=0:
            pad_len = 8 - (len(frames) % 8)
            frames.extend([frames[-1].copy() for _ in range(pad_len)])

        model.config.num_video_frames, model.config.fps = len(frames), 0
        model.config.llm_cfg['use_bfloat16'] = True

        return forward_pass(frames, model)"""
    
def load_frames_from_directory(video: str)->list:
    frames_paths = sorted([
                os.path.join(video, f) 
                for f in os.listdir(video) 
                if os.path.splitext(f)[1].lower() in [".jpg", ".png", ".jpeg"]
            ])
    frames = [Image.open(f).convert("RGB") for f in frames_paths]
    return frames

def pad_frames(frames: list)->list:
    if len(frames)%8 !=0:
        pad_len = 8 - (len(frames) % 8)
        frames.extend([frames[-1].copy() for _ in range(pad_len)])
    return frames

    
def triage_trauma(video: Union[list, str], model, f):

    frames = []
    if isinstance(video, list):
        frames = video
    elif os.path.isdir(video):
        frames = load_frames_from_directory(video)
    elif os.path.isfile(video):
        frames = extract_cropped_human_frames(video)
    else:
        raise ValueError("Invalid video input")
        
    if len(frames) == 0:
        raise ValueError("No frames found")
        
    # Pad frames to multiple of 8
    frames = pad_frames(frames)
    
    # Configure model
    model.config.num_video_frames, model.config.fps = len(frames), 0
    model.config.llm_cfg['use_bfloat16'] = True
    
    # Generate report
    if isinstance(f ,str):
        return forward_pass(frames, model, f)
    else:
        return forward_pass(frames, model)
        
    # except Exception as e:
    #     print(f"Error in triage_trauma: {str(e)}")
    #     return {}
    

def forward_pass(frames: list, model, video = "")-> dict:
    start_time = time.time()
    classes_trauma = ["normal", "wound", "amputation", "not testable"]
    classes_sh = ["present", "absent"]
    
    # Initialize generation config
    generation_config = model.default_generation_config
    generation_config = GenerationConfig(
        max_new_tokens = 1024,  # Will be updated based on context
        max_length = generation_config.max_length,
        pad_token_id = generation_config.pad_token_id,
        bos_token_id = generation_config.bos_token_id,
        eos_token_id = generation_config.eos_token_id,
        # do_sample=True,
        # temperature = 0.5,
    )

    prompt_ = prompts()
    system_prompt_thinking = prompt_.thinking()
    description = prompt_.description()
    Report = prompt_.report()
    Regen = prompt_.regen()

    description_sh = prompt_.description_sh()
    Report_sh = prompt_.report_sh()
    Regen_sh = prompt_.regen_sh()
    og_sh = prompt_.repiratory_severe()
    
    pattern = prompt_.regex()
    verify_amputation = prompt_.verify_amputation()

    pattern_sh = prompt_.regex_sh()

    

    ##DESCRIPTION GENERATION
    model.config.llm_cfg['max_length'] = 512
    model.config.llm_cfg['min_length'] = 10
    # model.config.llm_cfg['temperature'] = 0.1
    generation_config.max_new_tokens = 512  # For longer descriptions
    # generation_config.temperature=0.1
    use_thinking = True

    if use_thinking:
        prompt = system_prompt_thinking.format(question=description)
    else:
        prompt = description

    print("\n" + "="*80)
    print("INITIAL DESCRIPTION GENERATION")
    print("="*80)
    print("Prompt:")
    print("-"*40)
    print(prompt)
    print("-"*40)
    
    response_description = model.generate_content([prompt ,frames], generation_config=generation_config)
    print("\nModel Response:")
    print("-"*40)
    print(response_description)
    print("="*80)
    ##REGENERATING DESCRIPTION -> CHECK FOR REPEATING_PENALTY
    if check_tags(response_description)==False:
        # print("Tags not intact -> ", end="")

        model.config.llm_cfg['max_length'] = 512
        model.config.llm_cfg['min_length'] = 10
        # model.config.llm_cfg['temperature'] = 0.3
        generation_config.max_new_tokens = 512  # Keep same for regeneration
        # generation_config.temperature=0.3
        use_thinking = True

        if use_thinking: #FOUND AN ERROR description was getting entered twice in system prompt
            prompt = system_prompt_thinking.format(question=description)
        else:
            prompt = description
        response_description = model.generate_content([prompt,frames], generation_config=generation_config)
        print("regenerated response", response_description)
        # print("After regenerating: ", check_tags(response_description), end=" | ")
    
    else:
        # print("Tags intact -> ", end="")
        pass

    ##REPORT PROMPT
    model.config.llm_cfg['max_length'] = 256
    model.config.llm_cfg['min_length'] = 10
    # model.config.llm_cfg['temperature'] = 0.1
    generation_config.max_new_tokens = 256  # Shorter for report generation
    # generation_config.temperature=0.1
    use_thinking = False

    print("\n" + "="*80)
    print("TRAUMA REPORT GENERATION")
    print("="*80)
    print("Prompt:")
    print("-"*40)
    print("Video Description: " + response_description + " " + Report)
    print("-"*40)
    prompt_response_description = str(response_description).replace("<think>", "").replace("</think>", "").replace("<answer>", "").replace("</answer>", "")
    trauma_report = model.generate_content(["Video Description: " + prompt_response_description + " " + Report, frames], generation_config=generation_config)
    print("\nRaw Model Response:")
    print("-"*40)
    print(trauma_report)
    print("="*80)

    ##JSON VERFICATION
    trauma_report = trauma_report.lower().replace("\n", "").strip().replace('"', "").replace("'", "")
    print("formatted report", trauma_report)
    if bool(pattern.match(trauma_report)):
        # print("Valid format -> ", end="")
        pass

    else:
        model.config.llm_cfg['max_length'] = 256
        model.config.llm_cfg['min_length'] = 10
        # model.config.llm_cfg['temperature'] = 0.1
        use_thinking = False
        generation_config.max_new_token = 256
        # generation_config.temperature=0.1
        trauma_report = str(trauma_report).replace("<think>", "").replace("</think>", "").replace("<answer>", "").replace("</answer>", "")
        trauma_report = model.generate_content(["Previous Response: "+ trauma_report + Regen, frames], generation_config)
        print("Redone report", trauma_report)
        trauma_report = trauma_report.lower().replace("\n", "").strip().replace('"', "").replace("'", "")

        ##SO NOW WE EITHER BREAK WITHOUT GETTING A VALID REPORT OR WE NOW HAVE A VALID REPORT
        if bool(pattern.match(trauma_report)):
            # print("Valid after regen -> ", end="")
            pass
        else:
            print("\nFailed trauma report validation")
            print("Did not pass the check", trauma_report)
            trauma_report = "{head: _, torso: _, upper extremity: _, lower extremity: _}"
            # return {}
    
    ## Amputation - Wound verification
    

    # if use_thinking:
    #     verify = system_prompt_thinking.format(question=verify)

    trauma_report = clean_output(parse_to_dict(trauma_report), classes_trauma) #converting standard dictionary and cleaning

    if (trauma_report["upper extremity"] == "wound"  or trauma_report["upper extremity"] == "amputation") or (trauma_report["lower extremity"] == "wound" or trauma_report["upper extremity"] ==  "amputation"):
        model.config.llm_cfg['max_length'] = 256
        model.config.llm_cfg['min_length'] = 10
        model.config.llm_cfg['temperature'] = 0.1
        use_thinking = False
        generation_config.max_new_token = 256
        # generation_config.temperature=0.1

        print("\n" + "="*80)
        print("WOUND VERIFICATION")
        print("="*80)
        print("Prompt:")
        print("-"*40)
        print(str(trauma_report) + " " + verify_amputation)
        print("-"*40)
        prompt_trauma_report = str(trauma_report).replace("<think>", "").replace("</think>", "").replace("<answer>", "").replace("</answer>", "")
        trauma_report = model.generate_content([prompt_trauma_report +" "+ verify_amputation, frames], generation_config)
        print("\nModel Response:")
        print("-"*40)
        print(trauma_report)
        print("="*80)
        # print("Verified wounds -> ", end="")

        trauma_report = trauma_report.lower().replace("\n", "").strip().replace('"', "").replace("'", "")

        if bool(pattern.match(trauma_report)):
            # print("Valid format-wound", end=" | ")
            pass

        else:
            model.config.llm_cfg['max_length'] = 256
            model.config.llm_cfg['min_length'] = 10
            # model.config.llm_cfg['temperature'] = 0.1
            # generation_config.temperature=0.1
            use_thinking = False
            generation_config.max_new_token = 256

            trauma_report = model.generate_content(["Previous Response: "+ trauma_report + Regen, frames], generation_config)
            print("redone report", trauma_report)
            trauma_report = trauma_report.lower().replace("\n", "").strip().replace('"', "").replace("'", "")

            if bool(pattern.match(trauma_report)):
                # print("Valid after regen-wound", end=" | ")
                pass
            else:
                print("\nFailed wound verification")
                print("Did not pass the check", trauma_report)
                trauma_report = "{}"
            
        trauma_report = clean_output(parse_to_dict(trauma_report), classes_trauma)
    

    print("\n", end="")
    """##MOTOR ALERTNESS AND SEVERE HEMORRHAGE
    model.config.llm_cfg['max_length'] = 512
    model.config.llm_cfg['min_length'] = 10
    # model.config.llm_cfg['temperature'] = 0.1
    generation_config.max_new_tokens = 512  # Keep same for regeneration
    # generation_config.temperature=0.1
    use_thinking = True

    if use_thinking: #FOUND AN ERROR description was getting entered twice in system prompt
        prompt = system_prompt_thinking.format(question=description_sh)
    else:
        prompt = description_sh

    print("\n" + "="*80)
    print("MOTOR ALERTNESS DESCRIPTION")
    print("="*80)
    print("Prompt:")
    print("-"*40)
    print(prompt)
    print("-"*40)

    response_description = model.generate_content([prompt,frames], generation_config=generation_config)
    print("\nModel Response:")
    print("-"*40)
    print(response_description)
    print("="*80)
    # print("Motor Desc under tags?: ", check_tags(response_description), end=" | ")"""

    ##MOTOR REPORT PROMPT
    model.config.llm_cfg['max_length'] = 256
    model.config.llm_cfg['min_length'] = 10
    # model.config.llm_cfg['temperature'] = 0.1
    generation_config.max_new_tokens = 256  # Shorter for report generation
    # generation_config.temperature=0.1
    use_thinking = False

    print("\n" + "="*80)
    print("SEVERE HEMORRHAGE REPORT")
    print("="*80)
    print("Prompt:")
    print("-"*40)
    print(og_sh)
    print("-"*40)
    sevhem_ma_report = model.generate_content([og_sh, frames], generation_config=generation_config)
    
    # sevhem_ma_report = model.generate_content(["Video Description: " + response_description + " " + Report_sh, frames], generation_config=generation_config)
    print("\nModel Response:")
    print("-"*40)
    print(sevhem_ma_report)
    print("="*80)

    ##JSON VERFICATION
    sevhem_ma_report = sevhem_ma_report.lower().replace("\n", "").strip().replace('"', "").replace("'", "")

    if bool(pattern_sh.match(sevhem_ma_report)):
        # print("Valid format -> ", end="")
        pass

    else:
        model.config.llm_cfg['max_length'] = 256
        model.config.llm_cfg['min_length'] = 10
        # model.config.llm_cfg['temperature'] = 0.1
        use_thinking = False
        generation_config.max_new_token = 256
        # generation_config.temperature=0.1
        sevhem_ma_report = model.generate_content(["Previous Response: "+ sevhem_ma_report + Regen_sh, frames], generation_config)
        print("redone sev hem", sevhem_ma_report)
        sevhem_ma_report = sevhem_ma_report.lower().replace("\n", "").strip().replace('"', "").replace("'", "")

        ##SO NOW WE EITHER BREAK WITHOUT GETTING A VALID REPORT OR WE NOW HAVE A VALID REPORT
        if bool(pattern_sh.match(sevhem_ma_report)):
            # print("Valid after regen -> ", end="")
            pass
        else:
            print("\nFailed trauma report validation")
            print("Did not pass the check", sevhem_ma_report)
            sevhem_ma_report = "{motor alertness: _, severe hemorrhage: _}"

    
    sevhem_ma_report = clean_output(parse_to_dict(sevhem_ma_report), classes_sh)


    end_time = time.time()
    final_report = {**trauma_report, **sevhem_ma_report}
    final_report["Video"] = video
    final_report["time taken"] = f"{end_time - start_time:.2f}"
    print(final_report)
    print(f"\nTotal forward pass time: {end_time - start_time:.2f} seconds")
    new_df=pd.DataFrame([final_report])
    new_df.to_csv('./chirag_temp1_run4.csv',mode='a',index=False,header=False)

    return final_report



    
"""    print("\n", end="")  # New line before starting sevhem section
    
    model.config.llm_cfg['max_length'] = 512
    model.config.llm_cfg['min_length'] = 10
    model.config.llm_cfg['temperature'] = 0.1
    generation_config.max_new_tokens = 1024  # Longer for detailed assessment
    use_thinking = True

    if use_thinking:
        prompt = system_prompt_thinking.format(question=sevhem_ma)
    else:
        prompt = prompt

    sevhem_ma_report = model.generate_content([prompt, frames], generation_config=generation_config)

    ##JSON VERIFICATION
    model.config.llm_cfg['max_length'] = 256
    model.config.llm_cfg['min_length'] = 10
    model.config.llm_cfg['temperature'] = 0.1
    generation_config.max_new_token = 512
    use_thinking = False
    
    if bool(pattern.match(sevhem_ma_report.replace("\n", ""))):
        print("SevHem Valid format -> ", end="")

    else:
        sevhem_ma_report = model.generate_content(["Previous Response: "+ sevhem_ma_report + " " + Regen_sh, frames])
        if bool(pattern.match(sevhem_ma_report.replace("\n", ""))):
            print("SevHem Valid after regen -> ", end="")
        else:
            print("Failed sevhem validation")
            print("Did not pass the check", sevhem_ma_report)
            return {}
        
    classes=["Absent","Present","Normal","Abnormal"]
    sevhem_ma_report = clean_output(parse_to_dict(sevhem_ma_report), classes)

    end_time = time.time()
    print(f"\nTotal forward pass time: {end_time - start_time:.2f} seconds")"""
    



if __name__ == "__main__":
    model_path = "Efficient-Large-Model/LongVILA-R1-7B"
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
    video_dir = "/home/uasdtu/Documents/Chirag/Trauma-Darpa/casualty_videos/"
    save_dir = "chirag_run"
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

 
    
    # import pandas as pd
    # df = pd.read_csv('/home/uasdtu/Documents/cha0s/chirag_temp1_run3.csv',header=None)
    # video_names=list(df[6])


    
    print(f"Processing {len(os.listdir(video_dir))} videos")
    for idx, f in enumerate(os.listdir(video_dir)):
        # if f in video_names:
        #     continue

        print(f"Processing video: {f}")

        frames = extract_cropped_human_frames(video_dir+f, 16)
        # Create video-specific directory
        video_name = os.path.splitext(f)[0]  # Remove file extension
        video_dir_path = os.path.join(save_dir, video_name)
        os.makedirs(video_dir_path, exist_ok=True)
        
        # Save each frame
        for frame_idx, frame in enumerate(frames):
            save_path = os.path.join(video_dir_path, f"{frame_idx}.jpg")
            try:
                frame.save(save_path, "JPEG")
                print(f"Saved frame {frame_idx} to {save_path}")
            except Exception as e:
                print(f"Error saving frame {frame_idx} from video {f}: {str(e)}")
        print(triage_trauma(frames, model, f))
        print("\n\n\n")





    








