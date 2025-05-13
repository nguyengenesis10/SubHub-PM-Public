# SubHub Evals 

## Overview
Welcome to SubHub's eval repository. Building any good model requires a core understanding where the current opensource models are at, specifically in the context of construction. This mainly consists of good feature extraction, proper wording as expected by the industry, and eventually computing quantity takeoffs. 

## Description 
- ```feature_visualization``` : Initial attempt at mapping scopes to pages, nonsensical results.
- ```NHA_civil_model_outputs.csv``` : Spreadsheet with Qwen2.5-VL-72B outputs from the civil set using prompt from April 9th, see dev log, and max_token_count=500. 
- ```src``` : Source code to map human results to model outputs.

## To-do's
- [ ] Refactor feature extraction notebooks into seperate module.
- [ ] Finish extracting intermediate features and mapping to input image, to determine if visual realignment is required.
- [ ] Finish high level text output analysis to human generated scopes, focus on both semantic and literal accuray.  
