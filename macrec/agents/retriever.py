from typing import Any, List
from loguru import logger

from macrec.agents.base import ToolAgent
from macrec.tools import EmbeddingRetriever, InfoDatabase
from macrec.utils import read_json, get_rm


class Retriever(ToolAgent):
    """
    A simple tool agent that retrieves top-K candidate item ids for a given user id
    using precomputed embeddings. Optionally augments with item attributes from
    an info database.
    """

    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config)
        # align with other agents: instantiate an LLM and adopt its json_mode
        self.retriever_llm = self.get_LLM(config=config)
        self.json_mode = self.retriever_llm.json_mode
        # default K, but allow tool to override
        self.default_k: int = get_rm(config, 'top_k', 6)
        self.reset()

    @staticmethod
    def required_tools() -> dict[str, type]:
        # info_retriever is optional; we won't assert its presence strictly by design.
        # However, ToolAgent.validate_tools requires all to exist; to make optional,
        # only declare the candidate tool here and check for info at access time.
        return {
            'candidate_retriever': EmbeddingRetriever,
        }

    @property
    def candidate_retriever(self) -> EmbeddingRetriever:
        return self.tools['candidate_retriever']

    def _has_info_db(self) -> bool:
        return 'info_retriever' in self.tools and isinstance(self.tools['info_retriever'], InfoDatabase)

    def _format_candidates_with_attributes(self, ids: List[int]) -> str:
        if not self._has_info_db():
            return '\n'.join([f"{iid}: None" for iid in ids])
        info: InfoDatabase = self.tools['info_retriever']  # type: ignore[assignment]
        lines: List[str] = []
        for iid in ids:
            attr = info.item_info(item_id=int(iid))
            # item_info returns plain attributes or a prefixed string; normalize
            if attr.startswith('Item ') and ' Attributes:\n' in attr:
                attr = attr.split(' Attributes:\n', 1)[1]
            lines.append(f"{iid}: {attr}")
        return '\n'.join(lines)

    def forward(self, user_id: int, k: int | None = None) -> str:
        if not isinstance(user_id, int):
            return f"Invalid user id: {user_id}. It should be an integer."
        if k is None:
            k = self.default_k
        ids, _scores = self.candidate_retriever.retrieve(user_id=user_id, k=k)

        # ensure integer ids for downstream parsing when possible
        norm_ids: List[int] = []
        for tok in ids:
            try:
                norm_ids.append(int(tok))
            except Exception:
                # If non-integer token exists, let manager rank strings; set n_candidate accordingly
                pass

        # Expose n_candidate for answer parsing in ranking stage
        if hasattr(self.system, 'kwargs'):
            self.system.kwargs['n_candidate'] = len(ids)
            logger.debug(f'Set n_candidate={len(ids)} in system kwargs for rr task')
        else:
            logger.warning('System has no kwargs attribute, cannot set n_candidate')
            pass

        # Provide attributes if available for compatibility with prompts
        candidate_block = self._format_candidates_with_attributes([int(i) for i in ids])
        return f"Top {len(ids)} candidates for user {user_id} (format: id: attributes):\n{candidate_block}\n\nIMPORTANT: You must analyze ALL {len(ids)} items before using Finish. Items to analyze: {ids}"

    def invoke(self, argument: Any, json_mode: bool) -> str:
        if json_mode:
            # support {"type": "Retrieve", "content": [user_id, k]}
            if isinstance(argument, list):
                if len(argument) == 2:
                    user_id, k = argument
                elif len(argument) == 1:
                    user_id = argument[0]
                    k = None
                else:
                    return f"Invalid content for Retrieve: {argument}. Expect [user_id] or [user_id, k]."
            else:
                user_id = argument
                k = None
        else:
            # Expect "user_id" or "user_id, k"
            if isinstance(argument, str) and ',' in argument:
                try:
                    user_str, k_str = argument.split(',')
                    user_id = int(user_str.strip())
                    k = int(k_str.strip())
                except Exception:
                    return f"Invalid argument for Retrieve: {argument}. Expect 'user_id' or 'user_id, k'."
            else:
                try:
                    user_id = int(argument)
                    k = None
                except Exception:
                    return f"Invalid user id: {argument}."
        return self(user_id=user_id, k=k)


