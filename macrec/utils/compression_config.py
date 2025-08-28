"""Configuration utilities for prompt compression in MACRec agents."""

from typing import Dict, Any, Optional
from loguru import logger

def get_rm(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get value from config with default (local implementation to avoid circular imports)."""
    return config.get(key, default)

def configure_prompt_compression(
    agent_config: Dict[str, Any],
    default_compression_ratio: float = 0.5,
    default_threshold: int = 1000
) -> Dict[str, Any]:
    """Configure prompt compression settings for an agent.
    
    Args:
        agent_config: Agent configuration dictionary
        default_compression_ratio: Default compression ratio
        
    Returns:
        Dictionary with compression settings
    """
    compression_config = {
        'enable_compression': get_rm(agent_config, 'enable_compression', False),
        'compression_ratio': get_rm(agent_config, 'compression_ratio', default_compression_ratio)
    }
    
    return compression_config

def apply_compression_to_llm(llm, compression_config: Dict[str, Any]) -> None:
    """Apply compression configuration to an LLM instance.
    
    Args:
        llm: LLM instance (BaseLLM subclass)
        compression_config: Compression configuration dictionary
    """
    if hasattr(llm, 'enable_prompt_compression'):
        llm.enable_prompt_compression(
            enable=compression_config['enable_compression'],
            compression_ratio=compression_config['compression_ratio']
        )
        
        if compression_config['enable_compression']:
            logger.info(f"Applied compression to {llm.__class__.__name__}: "
                       f"ratio={compression_config['compression_ratio']}")
    else:
        logger.warning(f"LLM {llm.__class__.__name__} does not support compression")

def enable_compression_for_system(
    system,
    compression_ratio: float = 0.5,
    selective_agents: Optional[list] = None
) -> None:
    """Enable compression for all agents in a system.
    
    Args:
        system: System instance containing agents
        compression_ratio: Compression ratio to apply
        selective_agents: List of agent names to enable compression for. If None, enables for all.
    """
    compression_config = {
        'enable_compression': True,
        'compression_ratio': compression_ratio
    }
    
    # Handle manager agent
    if hasattr(system, 'manager') and system.manager:
        if selective_agents is None or 'manager' in selective_agents:
            if hasattr(system.manager, 'thought_llm'):
                apply_compression_to_llm(system.manager.thought_llm, compression_config)
            if hasattr(system.manager, 'action_llm'):
                apply_compression_to_llm(system.manager.action_llm, compression_config)
    
    # Handle other agents
    agent_attributes = ['analyst', 'reflector', 'searcher', 'interpreter', 'retriever']
    
    for attr_name in agent_attributes:
        if hasattr(system, attr_name):
            agent = getattr(system, attr_name)
            if agent and (selective_agents is None or attr_name in selective_agents):
                if hasattr(agent, 'llm'):
                    apply_compression_to_llm(agent.llm, compression_config)
                elif hasattr(agent, attr_name):  # For nested agent structures
                    nested_agent = getattr(agent, attr_name)
                    if hasattr(nested_agent, 'llm'):
                        apply_compression_to_llm(nested_agent.llm, compression_config)
    
    # Handle collaboration system with agents dict
    if hasattr(system, 'agents') and isinstance(system.agents, dict):
        for agent_name, agent in system.agents.items():
            if selective_agents is None or agent_name in selective_agents:
                if hasattr(agent, 'llm'):
                    apply_compression_to_llm(agent.llm, compression_config)
                elif hasattr(agent, 'thought_llm') and hasattr(agent, 'action_llm'):
                    apply_compression_to_llm(agent.thought_llm, compression_config)
                    apply_compression_to_llm(agent.action_llm, compression_config)
    
    logger.info(f"Applied compression to system agents: ,"
               f"selective={selective_agents}")

def get_compression_stats(system) -> Dict[str, Any]:
    """Get compression statistics from all agents in a system.
    
    Args:
        system: System instance containing agents
        
    Returns:
        Dictionary with compression statistics
    """
    stats = {
        'total_compressed_calls': 0,
        'total_token_savings': 0,
        'agents': {}
    }
    
    def collect_from_llm(llm, agent_name: str):
        if hasattr(llm, 'call_history'):
            compressed_calls = 0
            token_savings = 0
            
            for call in llm.call_history:
                compression_info = call.get('compression_info', {})
                if compression_info.get('compressed', False):
                    compressed_calls += 1
                    token_savings += compression_info.get('token_savings', 0)
            
            stats['agents'][agent_name] = {
                'compressed_calls': compressed_calls,
                'token_savings': token_savings,
                'total_calls': len(llm.call_history)
            }
            
            stats['total_compressed_calls'] += compressed_calls
            stats['total_token_savings'] += token_savings
    
    # Collect from manager
    if hasattr(system, 'manager') and system.manager:
        if hasattr(system.manager, 'thought_llm'):
            collect_from_llm(system.manager.thought_llm, 'manager_thought')
        if hasattr(system.manager, 'action_llm'):
            collect_from_llm(system.manager.action_llm, 'manager_action')
    
    # Collect from other agents
    agent_attributes = ['analyst', 'reflector', 'searcher', 'interpreter', 'retriever']
    
    for attr_name in agent_attributes:
        if hasattr(system, attr_name):
            agent = getattr(system, attr_name)
            if agent:
                if hasattr(agent, 'llm'):
                    collect_from_llm(agent.llm, attr_name)
                elif hasattr(agent, attr_name):
                    nested_agent = getattr(agent, attr_name)
                    if hasattr(nested_agent, 'llm'):
                        collect_from_llm(nested_agent.llm, attr_name)
    
    # Collect from collaboration system
    if hasattr(system, 'agents') and isinstance(system.agents, dict):
        for agent_name, agent in system.agents.items():
            if hasattr(agent, 'llm'):
                collect_from_llm(agent.llm, agent_name)
            elif hasattr(agent, 'thought_llm') and hasattr(agent, 'action_llm'):
                collect_from_llm(agent.thought_llm, f'{agent_name}_thought')
                collect_from_llm(agent.action_llm, f'{agent_name}_action')
    
    return stats
