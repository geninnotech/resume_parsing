import json

# def build_prompt(ocr_pdf_text):

#     with open("prompts/cv_structure.json", "r") as f:
#         json_structure=f.read()

#     with open("prompts/example.json", "r") as f:
#         example_json=json.dumps(json.load(f), indent=2)
    
#     with open("prompts/ocr_prompt.txt", "r") as f:
#         ocr_prompt=f.read()

#     prompt = f"""## json structure
# ```
# {json_structure}
# ```


# ## example json
# ```
# {example_json}
# ```

# ## ocr pdf text (text retrieved from pdf using tesseract)
# ```
# {ocr_pdf_text}
# ```


# {ocr_prompt}
# """

#     return prompt


import json

def build_prompt(ocr_pdf_text):

    with open("prompts/cv_structure.json", "r") as f:
        json_structure=f.read()

    with open("prompts/example.json", "r") as f:
        example_json=json.dumps(json.load(f), indent=2)
    
    with open("prompts/ocr_prompt.txt", "r") as f:
        ocr_prompt=f.read()

    prompt = f"""## json structure
```
{json_structure}
```


{ocr_prompt}
"""

    return prompt



def make_ocr_prompt():
    with open("prompts/simple_ocr.txt", "r") as f:
        return f.read()
    

def make_cv_prompt(resume_ocr):
    with open("prompts/cv_structure.json", "r") as f:
        json_structure= f.read()


    with open("prompts/structured_json.txt", "r") as f:
        prompt = f.read()
    
    prompt = prompt.replace("<##resume_ocr##>", resume_ocr)
    prompt = prompt.replace("<##structured_json##>", json_structure)

    return prompt
    
