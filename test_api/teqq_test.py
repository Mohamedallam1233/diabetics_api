import requests
import json
with open('mapping_dict.json', 'r') as json_file:
    mapping_dict = json.load(json_file)
cat_col = list(mapping_dict.keys())
cat_col = ["Age"] + cat_col[:-1]
sym_data = [70, "Male", "No", "Yes", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "No", "No", "No", "Yes", "No"]
payload = {k: v for k, v in zip(cat_col, sym_data)}
url = "http://127.0.0.1:8000/symptoms/"

response = requests.get(url)
csrftoken = response.cookies.get('csrftoken')
headers = {'Cookie': f'csrftoken={csrftoken}', 'X-CSRFToken': csrftoken}
response = requests.post(url, headers=headers, data=payload)
if response.status_code == 200:
    response_data = json.loads(response.text)
    message = response_data['message']
    print(f"prediction is a {message}")
else:
    print("POST request failed. Status code:", response.status_code)
