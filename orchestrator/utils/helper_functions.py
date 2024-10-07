import json
import re
from dataclasses import asdict, is_dataclass

from orchestrator.types.plan_execute_state import State


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


def save_state(state: State, filepath: str = 'state.txt') -> None:
    """
    Serializes the State object to JSON and writes it to a text file.

    Args:
        state (BaseModel): The state object to serialize.
        file_path (str): The path to the output text file.
    """
    try:
        state_json = state_to_json(state)
        # Write the JSON string to the file, overwriting existing content
        with open(filepath, 'w') as file:
            file.write(str(state_json))
        print(f'State Object saved to {filepath}')
    except Exception as e:
        print(f'Failed to save state: {e}')


def state_to_json(state) -> str:
    """
    Converts the State object into a JSON string, handling custom objects generically.

    Args:
        state (State): The State object to convert.

    Returns:
        str: A JSON string representation of the State object.
    """

    def convert_to_serializable(obj):
        if is_dataclass(obj):
            return {k: convert_to_serializable(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif hasattr(obj, '__dict__'):
            return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
        else:
            return (
                obj
                if isinstance(obj, (str, int, float, bool, type(None)))
                else str(obj)
            )

    # Convert the State object to a serializable dictionary
    serializable_state = convert_to_serializable(state)

    # Serialize the state to a JSON string
    state_json = json.dumps(serializable_state, indent=4)

    return state_json
