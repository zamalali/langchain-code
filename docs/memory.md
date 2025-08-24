<div align="center">
  <img src="../assets/logo.png" alt="LangCode Logo" width="180" />
  <h1><b>LangCode</b></h1>
  <p><b>The only CLI you need.</b></p>
</div>

# Memory

The memory module is responsible for managing the agent's memory and context. This includes storing past interactions, summarizing conversations, and providing relevant information to the agent when needed.

## Key Components

The memory module consists of the following key components:

- **Conversation History:** Stores the full history of interactions between the user and the agent. This includes both the user's input and the agent's responses. The conversation history is used to provide context for the agent's decisions and actions.
- **Summarization:** Summarizes long conversations to fit within the context window of the language model. This is important because language models have a limited context window, and long conversations can exceed this limit. Summarization helps to reduce the amount of text that needs to be processed by the language model, while still preserving the key information.
- **Entity Extraction:** Extracts key entities and concepts from the conversation to build a knowledge base. This allows the agent to learn from past interactions and to use this knowledge to improve its performance on future tasks.



- **Conversation History:** Stores the full history of interactions between the user and the agent.
- **Summarization:** Summarizes long conversations to fit within the context window of the language model.
- **Entity Extraction:** Extracts key entities and concepts from the conversation to build a knowledge base.

## Usage

The memory module is automatically integrated with the agent and does not require separate initialization. The agent will automatically store and retrieve information from the memory module as needed.

### Configuration

The memory module can be configured through the agent's settings. This allows you to customize the behavior of the memory module, such as the summarization method and the entity extraction method.

### Best Practices

- Keep conversations focused and concise.
- Use clear and specific language.
- Provide the agent with relevant context.



The memory module is automatically integrated with the agent and does not require separate initialization.
