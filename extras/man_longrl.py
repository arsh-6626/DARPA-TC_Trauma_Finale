from transformers import AutoModel, GenerationConfig
from prompts_chirag import prompts
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import re
import time
from PIL import Image




def single_video(video_path):
    
    prompt_ = prompts()
    system_prompt_thinking = prompt_.thinking()
    promptt2="""We are creating a experimental triage scenario where we are supposed to simulate a mass casualty incident consisting of casualty simulations made of actors and mannequins. Your task is to perform casualty triage under such simulation scenarios, Answer the following questions:
- Describe if the person present in the video is a real person or a mannequin ,assess on the basis of body movement and facial features?
Return all the answers in the <answer></answer> tag"""
    promptt = "Classify the casualty among the given categories - half body covered with blood, Normal "
    use_thinking = False
    start=time.time()
    print("Processing video: \n", video_path)

    generation_config = model.default_generation_config
    generation_config2 = GenerationConfig(
        max_new_tokens = 512,)
    
    if use_thinking:
        promptt = system_prompt_thinking.format(question=promptt)
  
    response_report = model.generate_content([promptt, {"path":video_path}],generation_config=generation_config2)#, {"path": video_path}])
    print(f"\nResponse: {response_report}")
    end= time.time()
    print(f"Time taken Completely: {end-start}")

if __name__ == "__main__":
    
    model_path = "Efficient-Large-Model/LongVILA-R1-7B"
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")

    #Model config
    model.config.num_video_frames, model.config.fps = 16, 0
    model.config.llm_cfg['use_bfloat16'] = True
    model.config.llm_cfg['max_length'] = 512
    model.config.llm_cfg['min_length'] = 10
    model.config.llm_cfg['temperature'] = 0.1
    dir_path = "/home/uasdtu/Documents/Chirag/Trauma-Darpa/casualty_videos"
    for i in os.listdir(dir_path):
        print(i)
        video_path= os.path.join(dir_path, i)
        single_video(video_path)
