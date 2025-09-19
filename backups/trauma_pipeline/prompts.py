import re

class prompts:
    def __init__(self):
        self.system_prompt_thinking = "You are a helpful assistant. The user asks a question, and then you solves it.\n\nPlease first think deeply about the question based on the given video, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {question}"
        self.initial_description_prompt = """We are creating a experimental triage scenario where we are supposed to simulate a mass casualty incident consisting of casualty simulations made of actors and mannequins. Your task is to perform casualty triage under such simulation scenarios, Answer the following questions:
- Describe and Locate any burn injuries on the casualty. 
- Describe if there are any injuries on the head ?
- Describe if the casualty is holding his injured body part with his hands?
- Describe if the casualty is wearing a helmet, or his head is occluded
- Describe if there are any wounds or blood on the chest of the casualty
- Describe if there are any wounds or blood on the abdomen of the casualty
- Describe if there is any wound or blood on the arms ?
- Describe if there is any sort of partial of complete amputation or removal of arms or forearms ?
- Describe if there is any wound or blood on the legs ?
- Describe if there is any sort of partial or complete amputation or removal of legs ?
Return all the answers in the <answer></answer> tag"""
        self.initial_report_prompt = """Consider the above given description and video, using that Analyze the casualty (actors and mannequins in video are all casualties) present in the video for one casualty only if multiples are visible focus on the biggest one and the one most visible in the video. Focus more on the casualty and take minor consideration of the environment.
Ignore the usage of camouflage clothes, nets, helmets, and other objects like bags, bottles and white-black boxes and create a report under the following assessment categories and criteria.


Evaluate each body region in these 4 categories under the following criteria for classification:
- normal: No visible injury.
- wound: visible injuries (bleeding, deformities, blood soaked clothing, gunshot wounds) or any Burn injuries.
- amputation: Severance or Removal of the limb.
- not testable : Area obscured or inaccessible, if the given body part is occluded and not visible due to external objects ( not clothing, clothing specifies as a part of body).

Classify them into the respective category:
- head(head, neck and face): [normal or wound or not testable]
- torso(shoulders, chest and abdominal region): [normal or wound or not testable]
- upper extremity(arms and hands): [normal or wound or amputation or not testable]]
- lower extremity(legs and feet): [normal or  wound or amputation or not testable]

Return the report as a dictionary STRICTLY IN THE FORMAT GIVEN BELOW:
{head:[<Class>],
torso:[<Class>],
upper extremity:[<Class>],
lower extremity:[<Class>]}
"""
        self.trauma_regeneration_report="""The following report was generated but is NOT in the required format:
Please correct this mistake.
Evaluate each body region in these 4 categories under the following criteria for classification:
- normal: No visible injury.
- wound: visible injuries (bleeding, deformities, blood soaked clothing, gunshot wounds) or any Burn injuries.
- amputation: Severance or Removal of the limb.
- not testable : Area obscured or inaccessible, if the given body part is occluded and not visible due to external objects ( not clothing, clothing specifies as a part of body).

Classify them into the respective category:
- head(head, neck and face): [normal or wound or not testable]
- torso(shoulders, chest and abdominal region): [normal or wound or not testable]
- upper extremity(arms and hands): [normal or wound or amputation or not testable]]
- lower extremity(legs and feet): [normal or  wound or amputation or not testable]

Return the report as a dictionary STRICTLY IN THE FORMAT GIVEN BELOW:
{head:[<Class>],
torso:[<Class>],
upper extremity:[<Class>],
lower extremity:[<Class>]}

Do not include any explanations, reasoning, or additional text. Only return the dictionary in the exact format shown above.
"""
        self.report_verify_amputation = ", Verify with respect to the video, if it shows severance of limbs or not? Classify severance as 'Amputation' and then return the correct report in the same format only."

        self.sevhem_ma_desc = """We are creating an experimental triage scenario where we simulate a mass casualty incident consisting of actors and mannequins as casualties. 

Your task is to carefully observe the given video and provide a **detailed descriptive narrative** for a **single casualty**. 
If multiple casualties are visible, focus on **the largest and most visible casualty**.

In your description, include detailed observations about:
- Body posture and movement (e.g., standing, walking, sitting, leaning, lying down).
- Is the posture upright or supported by either ground
- Visible blood or bleeding, including whether it appears active (oozing, squirting, pooling) or passive (dried blood or stains).
- Ignore camouflage clothing, nets, helmets, bags, bottles, white-black boxes, etc.

The goal is to provide enough detailed, structured observations that another model can later generate a triage report.

Return **only your descriptive narrative** inside the `<answer></answer>` tag."""

        self.sevhem_ma_report = """Using the description above and the video, analyze the casualty (actors and mannequins in the video are all casualties). 
If multiple casualties are visible, focus on the largest and most visible one. 
Prioritize the casualty over environmental factors.

Ignore camouflage clothes, nets, helmets, and unrelated objects such as bags, bottles, and white-black boxes.

---

Evaluate the casualty in the following **assessment categories**:

**motor alertness:**
- upright: Standing, Walking, or Sitting without support from external objects.
- supported: Leaning on an external object or lying on the ground.
- not testable: if the arms or legs are not visible or if they are immobile due to external factors (trapped under debris, tightly restrained, or obscured by objects).

**severe hemorrhage:**
- present: active bleeding (oozing, squirting, pooling) OR more than 50 percent of body surface covered in blood.
- absent: Otherwise.

---

**Return the report strictly in the following dictionary format:**
{motor alertness:[<Class>],
severe hemorrhage:[<Class>]}"""
        self.sevhem_ma_regen_report ="""The following report was generated but is NOT in the required format:
Please correct this mistake.

Evaluate the casualty in the following **assessment categories**:

**motor alertness:**
- upright: Standing, Walking, or Sitting without support from external objects.
- supported: Leaning on an external object or lying on the ground.
- not testable: if the arms or legs are not visible or if they are immobile due to external factors (trapped under debris, tightly restrained, or obscured by objects).

**severe hemorrhage:**
- present: active bleeding (oozing, squirting, pooling) OR more than 50 percent of body surface covered in blood.
- absent: Otherwise.

Re-analyze based on the given report and video, keeping the same assessment logic, but return the report STRICTLY in the following format only:
{motor alertness:[<Class>],
severe hemorrhage:[<Class>]}

Do not include any explanations, reasoning, or additional text. Only return the dictionary in the exact format shown above.
"""
    def thinking(self):
        return self.system_prompt_thinking

    def description(self):
        prompt= self.initial_description_prompt
        return prompt
    
    def report(self):
        prompt= self.initial_report_prompt
        return prompt
    
    def regen(self):
        prompt= self.trauma_regeneration_report
        return prompt
    
    def verify_amputation(self):
        return self.report_verify_amputation
    
    def description_sh(self):
        return self.sevhem_ma_desc
    
    def report_sh(self):
        prompt= self.sevhem_ma_report
        return prompt
    
    def regen_sh(self):
        return self.sevhem_ma_regen_report
    
    def regex(self):
        #Regex pattern
        pattern = re.compile(
            r'^\{\s*head\s*:\s*[^{}:]+\s*,\s*torso\s*:\s*[^{}:]+\s*,\s*upper\s+extremity\s*:\s*[^{}:]+\s*,\s*lower\s+extremity\s*:\s*[^{}:]+\s*,?\s*\}$',
            re.IGNORECASE  # Case-insensitive matching
        )   
        return pattern
    
    def regex_sh(self):
        pattern = re.compile(
            r'^\{\s*motor alertness\s*:\s*[^{}:]+\s*,\s*severe hemorrhage\s*:\s*[^{}:]+\s*,?\s*\}$',
            re.IGNORECASE  # Case-insensitive matching
        )   
        return pattern