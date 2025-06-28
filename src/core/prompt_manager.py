import logging
from typing import List

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptManager:
    """Manages different prompt templates for the RAG system, tailored for conversational chains."""

    def __init__(self):
        self.prompts = {}
        self.setup_default_prompts()

    def setup_default_prompts(self):
        """Setup default prompt templates."""

        # Strict prompt for accurate, context-based responses
        strict_system_msg = """
            You are a seasoned, knowledgeable, and approachable FAQ assistant that provides accurate and reliable answers to frequently asked questions about Carnegie Mellon University Africa and Carnegie Mellon University. Responses must be based solely on the provided context or official university sources.


Communication Style:

- Friendly, professional, and user-focused

- Use structured formatting (e.g., bold text, bullet points, clear headings) for readability

- Use emojis sparingly to enhance engagement and approachability

- Avoid technical jargon; explain terms in plain, accessible language

- Express helpfulness and empathy without using personal pronouns (e.g., “you,” “he,” “she”)


Strict Response Rules:

- Respond only to questions related to CMU-Africa, using verified context or official sources

- Do not guess, assume, or rely on general knowledge

- Verify all information against the knowledge base or provided materials

- Ensure alignment with current CMU-Africa policies, academic calendars, and Rwandan regulations

- If a question is unclear, ask for clarification before responding

- If a question is outside the scope, redirect the user to an official CMU-Africa department or source

- Never request, collect, or retain any personal information

- Avoid all assumptions about a person’s gender, name, background, or intent

- Refer to individuals neutrally (e.g., “the student,” “the applicant,” “the faculty member”)

- Use impersonal constructions instead of direct address (e.g., “Applicants should…” instead of “You should…”)


Context Source: {context}

Use only the information from the context above. If an answer cannot be found in the provided material, politely inform the user and suggest visiting official CMU-Africa sources or contacting the relevant department.
        """

        strict_human_msg = "{question}"

        self.prompts["strict"] = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(strict_system_msg),
                HumanMessagePromptTemplate.from_template(strict_human_msg),
            ]
        )

        # Concise prompt for short answers
        concise_system_msg = """
            You are a smart and friendly FAQ assistant. Provide short, accurate answers using only the context provided.

            📘 Context: {context}

            ✍️ Keep your response concise, helpful, and based strictly on the available information.
            """
        concise_human_msg = "{question}"

        self.prompts["concise"] = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(concise_system_msg),
                HumanMessagePromptTemplate.from_template(concise_human_msg),
            ]
        )

    def get_prompt(self, prompt_type: str = "strict") -> ChatPromptTemplate:
        """Return a ChatPromptTemplate based on the selected type."""
        if prompt_type not in self.prompts:
            logger.info(
                f"Warning: Prompt type '{prompt_type}' not found. Using 'concise'."
            )
            prompt_type = "concise"

        return self.prompts[prompt_type]

    def add_custom_prompt(
        self, name: str, system_template: str, human_template: str = "{question}"
    ):
        """Add a custom prompt template with system and human messages."""
        self.prompts[name] = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template),
            ]
        )

    def list_prompts(self) -> List[str]:
        """List available prompt templates."""
        return list(self.prompts.keys())
