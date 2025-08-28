"""Prompt compression utilities using LLM API for reducing token usage."""

import os
import json
from typing import Optional, Dict, Any
from loguru import logger
import pickle
import hashlib
from pathlib import Path

class APIPromptCompressor:
    """API-based prompt compressor using LLM services for reducing token usage."""
    
    def __init__(
        self,
        compression_ratio: float = 0.5,
        cache_dir: str = "cache/prompts",
        enable_cache: bool = True,
        min_compression_length: int = 200,
        preserve_structure: bool = True,
        llm_instance=None
    ):
        """Initialize the API-based prompt compressor.
        
        Args:
            compression_ratio: Target compression ratio (0.1-0.9)
            cache_dir: Directory to cache compressed prompts
            enable_cache: Whether to enable caching of compressed prompts
            min_compression_length: Minimum prompt length to trigger compression
            preserve_structure: Whether to preserve important structural elements
            llm_instance: LLM instance to use for compression (OpenRouter, Gemini, etc.)
        """
        self.compression_ratio = compression_ratio
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache
        self.min_compression_length = min_compression_length
        self.preserve_structure = preserve_structure
        self.llm_instance = llm_instance
        
        # Create cache directory
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Initialized API-based prompt compressor with ratio: {compression_ratio}")
    
    def _get_compression_prompt(self, text: str, ratio: float) -> str:
        """Generate the compression instruction prompt."""
        target_length = int(len(text) * ratio)
        
        return f"""You are an expert text compressor. Your task is to compress the following text while preserving all essential information, key details, and logical structure.

COMPRESSION GUIDELINES:
1. Target length: approximately {target_length} characters (compression ratio: {ratio:.1%})
2. Preserve all important facts, numbers, names, and key concepts
3. Maintain logical flow and structure
4. Remove redundant words and phrases
5. Use concise language while keeping clarity
6. Keep technical terms and domain-specific vocabulary

ORIGINAL TEXT:
{text}

COMPRESSED TEXT:"""

    def _get_cache_key(self, prompt: str, compression_ratio: float) -> str:
        """Generate cache key for a prompt."""
        content = f"{prompt}_{compression_ratio}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load compressed prompt from cache."""
        if not self.enable_cache:
            return None
            
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Save compressed prompt to cache."""
        if not self.enable_cache:
            return
            
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def _preserve_important_elements(self, prompt: str) -> tuple[str, list]:
        """Extract and preserve important structural elements."""
        preserved_elements = []
        modified_prompt = prompt
        
        if not self.preserve_structure:
            return modified_prompt, preserved_elements
        
        # Preserve JSON structure markers
        json_markers = [
            '{"type":',
            '"content":',
            '"action":',
            '"thought":',
            ']}',
            '[{',
            '}]'
        ]
        
        # Preserve action patterns
        action_patterns = [
            'Action:',
            'Thought:',
            'Observation:',
            'Input:',
            'Output:',
            'Finish[',
            'Search[',
            'Analyze['
        ]
        
        # Mark preserved elements (simple approach - could be enhanced)
        for i, marker in enumerate(json_markers + action_patterns):
            if marker in prompt:
                placeholder = f"__PRESERVE_{i}__"
                preserved_elements.append((placeholder, marker))
                modified_prompt = modified_prompt.replace(marker, placeholder)
        
        return modified_prompt, preserved_elements
    
    def _restore_preserved_elements(self, compressed_prompt: str, preserved_elements: list) -> str:
        """Restore preserved structural elements."""
        result = compressed_prompt
        for placeholder, original in preserved_elements:
            result = result.replace(placeholder, original)
        return result
    
    def compress_prompt(
        self,
        prompt: str,
        compression_ratio: Optional[float] = None,
        force_compression: bool = False
    ) -> Dict[str, Any]:
        """Compress a prompt using LLMLingua.
        
        Args:
            prompt: The prompt to compress
            compression_ratio: Override default compression ratio
            force_compression: Force compression even for short prompts
            
        Returns:
            Dictionary containing:
            - compressed_prompt: The compressed prompt
            - original_length: Original prompt length
            - compressed_length: Compressed prompt length
            - compression_ratio_actual: Actual compression ratio achieved
            - token_savings: Estimated token savings
            - from_cache: Whether result was loaded from cache
        """
        if not self.llm_instance:
            logger.warning("No LLM instance available for compression, returning original prompt")
            return {
                'compressed_prompt': prompt,
                'original_length': len(prompt),
                'compressed_length': len(prompt),
                'compression_ratio_actual': 1.0,
                'token_savings': 0,
                'from_cache': False,
                'compression_enabled': False
            }
        
        # Use provided ratio or default
        ratio = compression_ratio or self.compression_ratio
        
        # Skip compression for short prompts unless forced
        if len(prompt) < self.min_compression_length and not force_compression:
            logger.debug(f"Skipping compression for short prompt (length: {len(prompt)})")
            return {
                'compressed_prompt': prompt,
                'original_length': len(prompt),
                'compressed_length': len(prompt),
                'compression_ratio_actual': 1.0,
                'token_savings': 0,
                'from_cache': False,
                'compression_enabled': False
            }
        
        # Check cache first
        cache_key = self._get_cache_key(prompt, ratio)
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            cached_result['from_cache'] = True
            logger.debug(f"Loaded compressed prompt from cache (key: {cache_key[:8]}...)")
            return cached_result
        
        try:
            # Preserve important structural elements
            modified_prompt, preserved_elements = self._preserve_important_elements(prompt)
            
            # Compress the prompt using API
            compression_instruction = self._get_compression_prompt(modified_prompt, ratio)
            
            # Use the LLM to compress the prompt
            try:
                compressed_prompt = self.llm_instance(compression_instruction)
            except Exception as e:
                logger.error(f"LLM compression failed: {e}")
                return {
                    'compressed_prompt': prompt,
                    'original_length': len(prompt),
                    'compressed_length': len(prompt),
                    'compression_ratio_actual': 1.0,
                    'token_savings': 0,
                    'from_cache': False,
                    'compression_enabled': False
                }
            
            # Clean up the response (remove any instruction artifacts)
            compressed_prompt = compressed_prompt.strip()
            if compressed_prompt.startswith("COMPRESSED TEXT:"):
                compressed_prompt = compressed_prompt.replace("COMPRESSED TEXT:", "").strip()
            
            # Restore preserved elements
            compressed_prompt = self._restore_preserved_elements(compressed_prompt, preserved_elements)
            
            # Calculate metrics
            original_length = len(prompt)
            compressed_length = len(compressed_prompt)
            actual_ratio = compressed_length / original_length if original_length > 0 else 1.0
            
            # Estimate token savings (rough approximation: 1 token ≈ 4 characters)
            original_tokens = original_length // 4
            compressed_tokens = compressed_length // 4
            token_savings = original_tokens - compressed_tokens
            
            result = {
                'compressed_prompt': compressed_prompt,
                'original_length': original_length,
                'compressed_length': compressed_length,
                'compression_ratio_actual': actual_ratio,
                'token_savings': token_savings,
                'from_cache': False,
                'compression_enabled': True
            }
            
            # Save to cache
            self._save_to_cache(cache_key, result)
            
            logger.debug(f"Compressed prompt: {original_length} -> {compressed_length} chars "
                        f"({actual_ratio:.2f} ratio, ~{token_savings} token savings)")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to compress prompt: {e}")
            return {
                'compressed_prompt': prompt,
                'original_length': len(prompt),
                'compressed_length': len(prompt),
                'compression_ratio_actual': 1.0,
                'token_savings': 0,
                'from_cache': False,
                'compression_enabled': False,
                'error': str(e)
            }
    
    def compress_if_needed(
        self,
        prompt: str,
        max_tokens: int,
        tokens_per_char: float = 0.25
    ) -> Dict[str, Any]:
        """Compress prompt only if it exceeds token limit.
        
        Args:
            prompt: The prompt to potentially compress
            max_tokens: Maximum allowed tokens
            tokens_per_char: Estimated tokens per character ratio
            
        Returns:
            Compression result dictionary
        """
        estimated_tokens = len(prompt) * tokens_per_char
        
        if estimated_tokens <= max_tokens:
            return {
                'compressed_prompt': prompt,
                'original_length': len(prompt),
                'compressed_length': len(prompt),
                'compression_ratio_actual': 1.0,
                'token_savings': 0,
                'from_cache': False,
                'compression_enabled': False,
                'reason': 'within_token_limit'
            }
        
        # Calculate required compression ratio
        required_ratio = max_tokens / estimated_tokens
        # Add some buffer
        target_ratio = max(0.1, required_ratio * 0.8)
        
        logger.info(f"Prompt exceeds token limit ({estimated_tokens:.0f} > {max_tokens}), "
                   f"compressing with ratio {target_ratio:.2f}")
        
        return self.compress_prompt(prompt, compression_ratio=target_ratio)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        if not self.llm_instance:
            return {'compressor_available': False}
            
        cache_files = list(self.cache_dir.glob("*.pkl")) if self.cache_dir.exists() else []
        
        return {
            'compressor_available': True,
            'compression_ratio': self.compression_ratio,
            'min_compression_length': self.min_compression_length,
            'cache_enabled': self.enable_cache,
            'cache_size': len(cache_files),
            'cache_dir': str(self.cache_dir)
        }


# Global compressor instance
_global_compressor = None

def get_prompt_compressor(**kwargs) -> APIPromptCompressor:
    """Get or create global API-based prompt compressor instance."""
    global _global_compressor
    if _global_compressor is None:
        _global_compressor = APIPromptCompressor(**kwargs)
    return _global_compressor

def compress_prompt(prompt: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to compress a prompt using API."""
    compressor = get_prompt_compressor()
    return compressor.compress_prompt(prompt, **kwargs)

def compress_if_needed(prompt: str, max_tokens: int, **kwargs) -> Dict[str, Any]:
    """Convenience function to compress prompt if it exceeds token limit."""
    compressor = get_prompt_compressor()
    return compressor.compress_if_needed(prompt, max_tokens, **kwargs)
