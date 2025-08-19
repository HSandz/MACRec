from macrec.tools.base import Tool
from macrec.tools.summarize import TextSummarizer
from macrec.tools.wikipedia import Wikipedia
from macrec.tools.info_database import InfoDatabase
from macrec.tools.interaction import InteractionRetriever
from macrec.tools.embedding_retriever import EmbeddingRetriever

TOOL_MAP: dict[str, type] = {
    'summarize': TextSummarizer,
    'wikipedia': Wikipedia,
    'info': InfoDatabase,
    'interaction': InteractionRetriever,
    'embedding_retriever': EmbeddingRetriever,
}
