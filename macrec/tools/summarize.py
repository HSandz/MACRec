from loguru import logger

from macrec.tools.base import Tool
from macrec.llms import GeminiLLM
from macrec.utils import get_rm

class TextSummarizer(Tool):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Initialize Gemini LLM for summarization
        self.model_name: str = get_rm(self.config, 'model_name', 'gemini-2.0-flash')
        self.temperature: float = get_rm(self.config, 'temperature', 0.1)
        self.max_tokens: int = get_rm(self.config, 'max_tokens', 150)
        
        self.llm = GeminiLLM(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            json_mode=False
        )

    def reset(self) -> None:
        pass

    def summarize(self, text: str) -> str:
        """Summarize the given text using Gemini LLM.
        
        Args:
            text (str): The text to summarize.
            
        Returns:
            str: The summarized text.
        """
        try:
            prompt = f"""Please provide a concise summary of the following text in 1-2 sentences:

{text}

Summary:"""
            
            summary = self.llm(prompt)
            return f"Summarized text: {summary}"
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return f"Summarized text: {text[:100]}..."  # Fallback to truncation
