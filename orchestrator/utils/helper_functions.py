import json
import re


def extract_list(text):
    json_pattern = r'```(.*?)```'  # r'\{[\s\S]*?\}'
    match = re.search(json_pattern, text, re.DOTALL)

    if match:
        json_block = match.group(1)
        try:
            # Step 2: Parse the JSON block
            data = json.loads(json_block)

            # Step 3: Extract the 'list' object
            list_object = data.get('list', [])

            if list_object:
                # Set 4: Return list object
                return list_object
            else:
                print("The 'list' object is empty or not found in the JSON data.")
                return []

        except json.JSONDecodeError as e:
            print('Error decoding JSON:', e)
            return []
    else:
        print('No JSON block found in the provided text.')
        return []
