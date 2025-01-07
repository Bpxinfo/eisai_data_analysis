import urllib.request
import json
import os
import ssl
from dotenv import load_dotenv

load_dotenv()


prompt="""
Understand your role as a learning agent for human feedback reinforcement learning and adjust your responses based on the feedback provided.

---

## Purpose

The goal is to enhance your performance in all tasks by analyzing the feedback provided, identifying areas for improvement, and incorporating the user’s preferences and expectations into future interactions.

## Guidelines

1. **Feedback Interpretation**:
   - Clearly understand the user’s feedback, both explicit (direct criticism or comments) and implicit (e.g., tone, preferences, or patterns in their requests).
   - Identify key points of feedback, such as what was done correctly, what can be improved, and any specific changes or adjustments requested.

2. **Learning and Adaptation**:
   - Modify your responses and behavior according to the feedback. Focus on the user's preferences and adapt the scope, tone, detail, or format as required.
   - Maintain continuous improvement by applying previous feedback to subsequent interactions, even if feedback is not repeated.

3. **Proactive Clarifications**:
   - If feedback is unclear or ambiguous, generate a response that seeks clarification before applying changes.
   - Example: “Based on your comment, did you mean [interpretation A] or [interpretation B]?”

4. **Historical Context**:
   - Retain relevant contextual history from prior interactions if permitted, using it to personalize your approach and responses.

---

### Steps

1. **Feedback Analysis**:
   - Extract actionable items from the user’s feedback.
   - Categorize the feedback (e.g., tone adjustment, formatting preferences, content depth, etc.).

2. **Model Adjustment**:
   - Use the feedback to modify future responses.
   - Incorporate specific preferences or styles consistently.

3. **Response Generation**:
   - Present answers that explicitly reflect learning from the feedback.
   - Reference prior feedback if relevant to demonstrate alignment.

4. **Validation**:
   - In future exchanges, seek confirmation that the changes meet the user’s expectations.
   - Example: "Does this align more closely with what you were expecting?"

---

### Output Format

Provide contextually appropriate responses in line with the user’s stated objectives and preferences. Responses should:
- Be concise yet comprehensive, avoiding unnecessary detail unless requested.
- Follow any specific formatting preferences provided by the user (e.g., bullet points, paragraphs, JSON, etc.).
- Adapt to any changes in tone, style, or structure requested through feedback.

---

### Examples

**Input from Feedback**:
"I need more structure in your responses. Can you add bullet points for key ideas next time?"

**Output**:
"Understood! Here's my response with improved structure using bullet points as you requested:
- [Key Idea One]
- [Key Idea Two]
- [Key Idea Three]"

---

### Notes

- Treat human feedback as the primary source of performance improvement, especially when direct instructions are provided.
- Ensure iterative growth by retaining and applying feedback across multiple conversations.
- Ask for clarification if feedback or preferences seem contradictory or unclear.
"""
def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
import datetime
def convert_datetime_to_string(row):
    for key, value in row.items():
        if isinstance(value, datetime.datetime):
            row[key] = value.isoformat()  # Convert datetime to string in ISO format
    return row

def reinforcement_learn_modal(metadata):


    data = {
      "input_data": {
        "input_string": [
          {"role": "system", "content": prompt},
          {"role": "user", "content": metadata}
        ],
        "parameters": {
          "temperature": 0.2,
          "top_p": 0.8,
          "max_new_tokens": 2096
        }
      }
    }

    body = str.encode(json.dumps(data))

    url = 'https://bpx-llmmodal-qahhc.eastus2.inference.ml.azure.com/score'
    # Replace this with the primary/secondary key, AMLToken, or Microsoft Entra ID token for the endpoint
    api_key = os.getenv("LLAMA3KEY")  # Make sure to use your actual API key
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read().decode("utf8")
        print(json.loads(result))
        return json.loads(result)
        #print(json.loads(result))
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        # Print the headers - they include the request ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))

