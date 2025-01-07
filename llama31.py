import urllib.request
import json
import os
import ssl
from dotenv import load_dotenv

load_dotenv()


prompt="""
You are an AI assistant tasked with analyzing customer feedback. Your job is to Extract keywords, relationships, synonyms, sentiments, and details related to safety, diagnosis, and provide the output in JSON format."
**Output JSON:**

```json
{
  "result": [
    {
      "keywords": ["community Summit", "pioneers", "patients", "caregivers", "Eisai", "community leaders"],
      "Theme": ["pioneering healthcare initiatives"],
      "safety": [],
      "diagnose": [],
      "treatment": [],
      "synonyms": ["gathering", "initiators", "wellbeing seekers", "caretakers", "company", "leaders"],
      "sentiment": "8",
      "nodes": [
        { "id": "pioneers", "group": 1, "label": "pioneers" },
        { "id": "community Summit", "group": 1, "label": "community Summit" },
        { "id": "Eisai", "group": 2, "label": "Eisai" },
        { "id": "community leaders", "group": 2, "label": "community leaders" }
      ],
      "links": [
        { "source": "pioneers", "target": "community Summit", "relationship": "hosts" },
        { "source": "community Summit", "target": "Eisai", "relationship": "initiated by" },
        { "source": "Eisai", "target": "community leaders", "relationship": "invited by" }
      ],
      "AnalyzeThoroughly": "The statement emphasizes leadership in healthcare initiatives, hinting at active future community involvement.",
      "THEME": "Pioneering community healthcare engagement",
      "ISSUE": "Delayed action in community healthcare initiatives"
    }
  ]
}
```
"""

def chat(user):
    def allowSelfSignedHttps(allowed):
        # bypass the server certificate verification on client side
        if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
            ssl._create_default_https_context = ssl._create_unverified_context

    allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

    # Request data goes here
    # More information can be found here:
    # https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
    data = {
      "input_data": {
        "input_string": [
          {"role": "system", "content": prompt},
          {"role": "user", "content": user}
        ],
        "parameters": {
          "temperature": 0.8,
          "top_p": 0.8,
          "max_new_tokens": 2096
        }
      }
    }

    body = str.encode(json.dumps(data))

    url = 'https://bpx-llmmodal-qahhc.eastus2.inference.ml.azure.com/score'
    # Replace this with the primary/secondary key, AMLToken, or Microsoft Entra ID token for the endpoint
    api_key = os.getenv("LLAMA3KEY") # Make sure to use your actual API key
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read().decode("utf8")
        #print(json.loads(result))
        return result
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the request ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))
