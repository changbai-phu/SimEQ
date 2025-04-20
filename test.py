from google import genai
from google.genai import types
from IPython.display import Markdown
# genai.__version__

import config.config as cfg

print(cfg.DISASTER_REPORT_PATH) 
print(cfg.document_sim_file)

# Example
location = 'Los Angeles'
magnitude = 6.8
population_density = 'high'
time = '03:45 AM'

GOOGLE_API_KEY = ""
client = genai.Client(api_key=GOOGLE_API_KEY)

file = client.files.upload(file=cfg.document_sim_file)

# Example query to test if everything works
'''
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=["What are the impacts of a 7.0 magnitude earthquake?", file])
print(response.text)
'''

# Simulate EQ scenarios based on given variables and documents
def sim_scenario():
    request = f"""
    You're a crisis simulator. Create a realistic earthquake scenario based on:
    Location: {location}
    Magnitude: {magnitude}
    Population Density: {population_density}
    Time: {time}
    """
    model_config = types.GenerateContentConfig(temperature=0.1, top_p=0.95)

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        config=model_config,
        contents=[request, file]) 

    # final_resp = response.text
    # print(final_resp)
    return response.text
    

resp = sim_scenario()
print("*****", resp)
# Markdown(resp)