import asyncio
import os
from typing import Any, Dict

import openai
from dotenv import load_dotenv


class CodeAssistant:
    def __init__(self):
        """
        Initializes the Code_Assistant with the provided OpenAI client.

        :param client: An instance of the OpenAI client.
        """
        load_dotenv()

        api_key = os.getenv('OPENAI_API_KEY')

        self.client = openai.OpenAI(api_key=api_key)
        # Create an assistant when initializing the class
        self.assistant = self.client.beta.assistants.create(
            name='Code Interpreter',
            instructions='You are a dedicated assistant that devises and executes code.',
            model='gpt-4o-mini',
            tools=[{'type': 'code_interpreter'}],
        )

    async def execute_message(self, message: Dict[str, Any]) -> Any:
        """
        Processes the message object, executes code if required, and returns the output.

        :param message: A dictionary representing the message object.
        :return: The output from the executed code.
        """
        # Create a new thread
        thread = self.client.beta.threads.create()

        # Add the user's message to the thread
        self.client.beta.threads.messages.create(
            thread_id=thread.id, role='user', content=message['content']
        )

        # Run the assistant
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id, assistant_id=self.assistant.id
        )

        # Wait for the run to complete
        while run.status not in ['completed', 'failed']:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=run.id
            )
            await asyncio.sleep(1)  # Wait for 1 second before checking again

        if run.status == 'failed':
            return f'Run failed with error: {run.last_error}'

        # Retrieve the messages
        messages = self.client.beta.threads.messages.list(thread_id=thread.id)

        # Return the assistant's response
        for message in messages:
            if message.role == 'assistant':
                return message.content[0].text.value

        return 'No response from assistant.'
