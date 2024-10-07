import os

from groq import Groq


class Orchestrator:
    def __init__(self, model_name='llama3-70b-8192', temperature=0.5, max_tokens=1024):
        self.client = Groq(
            api_key=os.environ.get('GROQ_API_KEY'),
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = 0.1

    def process_request(self, system_message):
        # Define the prompt template
        prompt_template = """
        Review this content and tell me if the task issued in the field "system_task" has been successfully completed. 
        If you can't find any evidence that it has been successfully completed, assume it has not. 
        If you find the tasks have not been completed, provide a list of next steps that will complete the given task. 
        Provide these tasks as a simple list in the object "list" and in .json format with triple backticks. 
        If code is needed to execute any of these steps, indicate that code is needed by returning `code_needed = True` as part of the specific step description.
        Here is the CONTENT:
        {system_message}
        """

        # Format the prompt with the user message
        formatted_prompt = prompt_template.format(system_message=system_message)

        # Create a chat completion request
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    'role': 'system',
                    'content': 'You orchestrate multi-agent interactions. Be precise in your output and clearly follow the instructions provided.',
                },
                {'role': 'user', 'content': formatted_prompt},
            ],
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stop=None,
            stream=False,
        )

        # Return the response content
        return chat_completion.choices[0].message.content
