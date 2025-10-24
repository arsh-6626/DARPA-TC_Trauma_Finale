# UAS DTU - DARPA Triage Challenge - Trauma Pipeline.

The given repository was used in DARPA Triage Challenge - Phase 2, in October 2025 by team UAS-DTU, where we won 150k USD as runner's up in self funded category(Systems). The repository was evaluating `Trauma` and `Severe Hemmorhage` categories

## File Structure

* `/backups/final_act_v2/`: scripts used during final ablations
* `/extras`: extra scripts like qna, etc/
* `/reports`: All generated reports and conf matrices
* `prompts_final.py`: collection of final prompts
* `the_final_act.py`: Final Script (Untested yet)
* `utils.py`: all helper functions


## Process Flow
<p align="center">
<img height="1044" alt="Untitled Diagram drawio (2)" src="https://github.com/user-attachments/assets/9779a80e-5d03-4521-b1b0-802f3c5de939" />
</p>

## Results and Evaluation

### Results accros all of the DARPA Provided Dataset
```
======================================
      Category-wise Accuracy          
======================================
Accuracy for 'Head': 80.00% (40 / 50 correct)
Accuracy for 'Torso': 78.00% (39 / 50 correct)
Accuracy for 'Upper Extremity': 74.00% (37 / 50 correct)
Accuracy for 'Lower Extremity': 76.00% (38 / 50 correct)
Accuracy for 'Severe Hemorrhage': 74.00% (37 / 50 correct)

======================================
          Total Accuracy              
======================================
Overall Accuracy: 76.40% (191 / 250 correct)
```

### Confusion Matrices
___________
<p align="center">
  <img src="https://github.com/user-attachments/assets/126e2080-ad72-49cb-bb0e-40071405434f" width=250">
  <img src="https://github.com/user-attachments/assets/5efd6fbd-84d2-4838-a606-ea35d69ef5d0" width="250">
  <img src="https://github.com/user-attachments/assets/fc2e7e19-4c6c-465a-be5a-4be5bc648217" width="250">
  <img src="https://github.com/user-attachments/assets/de9f6400-b3d7-4557-98ab-ddb29c413d5c" width="250">
  <img src="https://github.com/user-attachments/assets/9f76cd41-03fe-4531-99d4-5ba260a78257" width="250">
</p>



