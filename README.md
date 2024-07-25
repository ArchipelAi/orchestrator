# Orchestrator
This orchestrator is some wild multi agent stuff... Oo!

## How to install the project
1. clone the repo locally
```bash
$ git clone https://github.com/ArchipelAi/orchestrator.git
```
2. go into the project
```bash
$ cd orchestrator
```
3. Install [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
4. Install project dependencies
```
$ poetry install
```

## How to run the project
1. create a copy of the .env.example file
2. rename the copy to .env
3. change the api key for OPENAI_API_KEY to your key
4. run to execute a test run for the planner
```bash
$ poetry run planner
```
