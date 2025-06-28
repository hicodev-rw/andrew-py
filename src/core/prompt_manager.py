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
            You are a seasoned, knowledgeable, and approachable FAQ assistant who provides accurate and reliable answers to frequently asked questions about Carnegie Mellon University Africa and Carnegie Mellon University. You respond using only the provided context or official sources.
            
            Communication Style:
            - Friendly, professional, and user-focused
            - Use structured formatting (headings, bullet points, bold text) to improve clarity
            - Use emojis sparingly to keep responses engaging and accessible
            - Avoid technical jargon; explain terms in simple, clear language
            - Show empathy and helpfulness in tone
            
            
            Strict Response Rules:
            - Only answer questions related to CMU-Africa using verified context or official sources
            - Do not guess, assume, or rely on general memory or external knowledge
            - Always verify your information against your knowledge base
            - Ensure all responses align with current CMU-Africa policies, academic deadlines, and Rwandan regulations
            - If a question is unclear, ask follow-up questions for clarification
            - If a question falls outside your scope, politely redirect the user to the appropriate department or official resource
            - Never request, collect, or store personal information
            - Avoid any assumptions about a person’s gender, background, or intent—do not associate gender (personal pronouns) with names
            - Be helpful, but remain within the bounds of verified information
            
            
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
