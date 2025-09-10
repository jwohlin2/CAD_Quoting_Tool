import json
from feature_extractor import analyze_cad_file

def create_llm_prompt(geometric_data):
    """Formats the prompt for the LLM."""
    
    # Convert the geometric data dictionary to a formatted string
    data_string = json.dumps(geometric_data, indent=4)
    
    # This is our prompt template from Step 2
    prompt = f"""
# ROLE:
You are an expert manufacturing estimator for a high-precision machine shop. Your task is to analyze geometric data extracted from a CAD file and make initial recommendations for quoting variables.

# CONTEXT:
A CAD file has been analyzed, and the following geometric data was extracted. All dimensions are in millimeters.

{data_string}

# INSTRUCTIONS:
Based on the geometric data provided, fill in the following JSON object with your best estimates. Follow these rules:
- If the Smallest_Internal_Radius_mm is less than 0.2, Sinker_EDM_Required must be "True".
- If the Heuristic_Complexity_Score is over 150, set the Tolerance_Profile to "Tight". Otherwise, set it to "Standard".
- If the Heuristic_Complexity_Score is over 250, set Custom_Fixture_Required to "Yes".
- If any holes are detected, assume Live_Tooling_Required is "True".
- If the Smallest_Internal_Radius_mm is "Not Found", assume it's a simple part that doesn't need EDM.
- For all other fields, use your expertise to choose the most likely default option.

# OUTPUT (JSON format only):
{{
  "PM-01_Quote_Priority": "Standard",
  "MIL-02_Number_of_Milling_Setups": "2",
  "MIL-12_Thin_Wall_Features_Present": "False",
  "TRN-03_Live_Tooling_Required": "True",
  "GRD-08_Sinker_EDM_Required": "False",
  "FIN-01_Manual_Deburring_Level": "Standard Edge Break",
  "ASM-04_Precision_Fitting_Required": "False",
  "QC-05_FAIR_Report_Required": "False",
  "ENG-05_Custom_Fixture_Required": "No",
  "GEO-05_Tolerance_Profile": "Standard (+/- 0.1mm)",
  "GEO-07_Critical_Surface_Finish_Ra": "3.2"
}}
"""
    return prompt

def call_llm_api(prompt):
    """
    *** This is a placeholder function! ***
    In a real application, you would put your code here to call an LLM API
    (like Google's Gemini API or OpenAI's GPT API). You would send the
    'prompt' string and get back a JSON string.
    
    For this example, we will simulate the LLM's response.
    """
    print("\n--- Sending the following prompt to the LLM (simulation) ---")
    print(prompt)
    
    # --- SIMULATED LLM RESPONSE ---
    # This is what the LLM would send back after analyzing the prompt.
    simulated_response = """
{
  "PM-01_Quote_Priority": "Standard",
  "MIL-02_Number_of_Milling_Setups": "3",
  "MIL-12_Thin_Wall_Features_Present": "False",
  "TRN-03_Live_Tooling_Required": "True",
  "GRD-08_Sinker_EDM_Required": "True",
  "FIN-01_Manual_Deburring_Level": "Standard Edge Break",
  "ASM-04_Precision_Fitting_Required": "False",
  "QC-05_FAIR_Report_Required": "False",
  "ENG-05_Custom_Fixture_Required": "Yes",
  "GEO-05_Tolerance_Profile": "Tight (+/- 0.025mm)",
  "GEO-07_Critical_Surface_Finish_Ra": "1.6"
}
"""
    return json.loads(simulated_response)


# --- Main Workflow ---
if __name__ == "__main__":
    # 1. Analyze the CAD file to get the geometric data
    cad_file = "your_part.step"  # <-- IMPORTANT: Change this to your file
    geometric_features = analyze_cad_file(cad_file)

    if geometric_features:
        # 2. Create the prompt for the LLM
        llm_prompt = create_llm_prompt(geometric_features)
        
        # 3. Call the LLM to get the filled-out variables
        quote_variables = call_llm_api(llm_prompt)
        
        # 4. Display the final result!
        print("\n\n--- LLM-Powered Quote Variables ---")
        print(json.dumps(quote_variables, indent=4))
        print("---------------------------------")
        print("\nThese variables are now ready to be plugged into your quoting sheet!")
