import re

class prompts:
    def __init__(self):
        self.system_prompt_thinking = "You are a helpful assistant. The user asks a question, and then you solves it.\n\nPlease first think deeply about the question based on the given video, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {question}"
        self.description_p = """We are creating a experimental triage scenario where we are supposed to simulate a mass casualty incident consisting of casualty simulations made of actors and mannequins. Your task is to perform casualty triage under such simulation scenarios, Answer the following questions:
- Describe if there are any injuries on the head ?
- Describe if the casualty is wearing a helmet, or his head is occluded
- Describe if there are any wounds or blood on the CHEST of the casualty
- Describe if there are any wounds or blood on the ABDOMEN of the casualty
- Is there any wound or blood on the ARMS ?
- Is there any sort of partial of complete amputation or removal of ARMS or FOREARMS ?
- Is there any wound or blood on the LEGS ?
- Is there any sort of partial or complete amputation or removal of LEGS ?
Return all the answers in the <answer></answer> tag"""
        self.description_amp = """just print yes"""
        self.report_p = """Consider the above given descriptiona and video, using that Analyze the casualty(actors and mannequins in video are all casualties) present in the video for one casualty only if multiples are visible focus on the biggest one and the one most visible in the video. Focus more on the casualty and take minor consideration of the environment.
Ignore the usage of camouflage clothes, nets, helmets, and other objects like bags, bottles and white-black boxes and create a report under the following assessment categories and criteria.


Evaluate each body region in these 4 categories under the following criteria for classification:
- Normal: No visible injury.
- Wound: Visible injuries (bleeding, burns, deformities, blood-soaked clothing, gunshot wounds).
- Amputation: Severance or Removal of the limb.
- Not Testable (NT): Area obscured or inaccessible, if the given body part is occluded and not visible due to external objects ( not clothing, clothing specifies as a part of body).

Classify them into the respective category:
- Head(Head, Neck and Face): [Normal or Wound or Not Testable]
- Torso(Shoulders, Chest and Abdominal Region): [Normal or Wound or Not Testable]
- Upper Extremity(Arms and Hands): [Normal or Wound or Amputation or Not Testable]]
- Lower Extremity(Legs and Feet): [Normal or  Wound or Amputation or Not Testable]

Return the report as a dictionary STRICTLY IN THE FORMAT GIVEN BELOW:
{Head:[<Class>],
Torso:[<Class>],
Upper Extremity:[<Class>],
Lower Extremity:[<Class>]}
"""
        self.report_verify = ", Verify with respect to the video, if it shows severance of limbs or not? Classify severance as 'Amputation' and then return the correct report in the same format only."

        self.regen_p="""The following report was generated but is NOT in the required format:
Please correct this mistake.
Evaluate each body region in these 4 categories under the following criteria for classification:
- Normal: No visible injury.
- Wound: Visible injuries (bleeding, burns, deformities, blood-soaked clothing, gunshot wounds).
- Amputation: Severance or Removal of the limb.
- Not Testable (NT): Area obscured or inaccessible, if the given body part is occluded and not visible due to external objects ( not clothing, clothing specifies as a part of body).

Classify them into the respective category:
- Head(Head, Neck and Face): [Normal or Wound or Not Testable]
- Torso(Shoulders, Chest and Abdominal Region): [Normal or Wound or Not Testable]
- Upper Extremity(Arms and Hands): [Normal or Wound or Amputation or Not Testable]]
- Lower Extremity(Legs and Feet): [Normal or  Wound or Amputation or Not Testable]

Re-analyze based on the given report and video, keeping the same assessment logic, but return the report STRICTLY in the following format only:
{Head:[<Class>],
Torso:[<Class>],
Upper Extremity:[<Class>],
Lower Extremity:[<Class>]}

Do not include any explanations, reasoning, or additional text. Only return the dictionary in the exact format shown above.
"""
        self.resp= """
Consider the above given descriptiona and video, using that Analyze the casualty(actors and mannequins in video are all casualties) present in the video for one casualty only if multiples are visible focus on the biggest one and the one most visible in the video. Focus more on the casualty and take minor consideration of the environment.
Ignore the usage of camouflage clothes, nets, helmets, and other objects like bags, bottles and white-black boxes and create a report under the following assessment categories and criteria.

Evaluate each body region in these 2 categories under the following criteria for classification:
MOTOR ALERTNESS:
- Normal - Standing, Walking or Sitting without Support
- Abnormal - Leaning on some external object, lying down on ground

RESPIRATORY DISTRESS:
- Present - In tripod position ( leaning forward with your hands on your knees, looking like having difficulty to breather - posture adopted during respiratory distress where a person sits or stands, leans forward, and supports their upper body by bracing their hands on their knees or a surface)
- Absent - In some other position

SEVERE HEMORRHAGE
- Present - Active bleeding (oozing, squirting, pooling), more than 50 percent of the body surface covered in blood.
- Absent: otherwise.
"""

    def thinking(self):
        prompt= self.system_prompt_thinking
        return prompt
    def description(self):
        prompt= self.description_p
        return prompt
    def report(self):
        prompt= self.report_p
        return prompt
    def regen(self):
        prompt= self.regen_p
        return prompt
    def repiratory_severe(self):
        prompt= self.resp
        return prompt
    def regex(self):
        #Regex pattern
        pattern = re.compile(
            r'^\{Head: [A-Za-z]+,\s*'
            r'Torso: [A-Za-z]+,\s*'
            r'Upper Extremity: [A-Za-z]+,\s*'
            r'Lower Extremity: [A-Za-z]+\}$'
        )
        return pattern
    def verify(self):
        return self.report_verify
    
if __name__=='__main__':
    prompt = prompts()
  
    # print(prompt.repiratory_severe())
    pattern = prompt.regex()
    response_report = """{Head: Normal,
Torso: Normal,
Upper Extremity: Normal,
Lower Extremity: Normal}"""
    print(response_report)
    print(pattern.match(response_report.replace("\n", "")))