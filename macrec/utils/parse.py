# Description: This file contains functions for parsing agent actions and answers.

import re
import json
from typing import Any

def parse_action(action: str, json_mode: bool = False) -> tuple[str, Any]:
    """Parse agent action.

    Args:
        `action` (`str`): Agent action in string format.
        `json_mode` (`bool`, optional): Whether the action is in JSON format. Defaults to `False`.
    Returns:
        `tuple[str, Any]`: Action type and argument.
    """
    if json_mode:
        try:
            # Clean the action string to handle multi-line responses
            action = action.strip()
            
            # Clean up common LLM escaping mistakes before parsing
            # LLMs sometimes incorrectly escape $ and other characters that don't need escaping in JSON
            # Use regular strings (not raw strings) to match actual escape sequences
            action_cleaned = action.replace('\\$', '$').replace('\\#', '#').replace('\\%', '%').replace('\\&', '&')
            
            # Try to parse the action directly first (handles most cases)
            try:
                json_action = json.loads(action_cleaned)
                # Handle case where action is wrapped in an array (common LLM mistake)
                if isinstance(json_action, list) and len(json_action) == 1:
                    json_action = json_action[0]
                
                # Proceed to validation
                if 'type' not in json_action:
                    return 'Invalid', None
                
                # Validate command type and content
                action_type = json_action['type']
                valid_types = ['Analyse', 'UserInfo', 'ItemInfo', 'UserHistory', 'ItemHistory', 'Finish']
                
                # Convert action_type to lowercase for comparison, but find the correct case from valid_types
                action_type_lower = action_type.lower()
                valid_action = None
                for valid_type in valid_types:
                    if valid_type.lower() == action_type_lower:
                        valid_action = valid_type
                        break
                
                if valid_action is None:
                    return 'Invalid', None
                
                # Special validation: Finish command must have content
                content = json_action.get('content', None)
                if valid_action == 'Finish' and (content is None or content == ""):
                    return 'Invalid', None
                
                return valid_action, content
                
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from mixed content
                pass
            
            # If action contains multiple lines, try to extract just the JSON part
            if '\n' in action:
                lines = action.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        action = line
                        break
                else:
                    # If no complete JSON line found, try to reconstruct the JSON
                    # Look for the start and end of JSON across multiple lines
                    json_content = ""
                    in_json = False
                    for line in lines:
                        line = line.strip()
                        if line.startswith('{'):
                            in_json = True
                            json_content = line
                        elif in_json:
                            json_content += " " + line
                            if line.endswith('}'):
                                action = json_content
                                break
            
            # Handle cases where the response contains text before/after the JSON
            # Look for JSON object in the string using proper brace matching
            import re
            
            # First try to find a complete JSON object using brace counting
            start_idx = action.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = -1
                in_string = False
                escape_next = False
                
                for i, char in enumerate(action[start_idx:], start_idx):
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        continue
                    
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                
                if end_idx != -1:
                    action = action[start_idx:end_idx]
                else:
                    # If brace counting fails, try to parse the entire action as JSON
                    # This handles cases where the JSON might be well-formed but complex
                    pass
            
            # Apply same cleaning to extracted JSON
            action_cleaned = action.replace('\\$', '$').replace('\\#', '#').replace('\\%', '%').replace('\\&', '&')
            json_action = json.loads(action_cleaned)
            # Handle case where action is wrapped in an array (common LLM mistake)
            if isinstance(json_action, list) and len(json_action) == 1:
                json_action = json_action[0]
            
            # Ensure required fields exist
            if 'type' not in json_action:
                return 'Invalid', None
            
            # Validate command type (case-insensitive)
            action_type = json_action['type']
            valid_types = ['Analyse', 'UserInfo', 'ItemInfo', 'UserHistory', 'ItemHistory', 'Finish']
            
            # Convert action_type to lowercase for comparison, but find the correct case from valid_types
            action_type_lower = action_type.lower()
            valid_action = None
            for valid_type in valid_types:
                if valid_type.lower() == action_type_lower:
                    valid_action = valid_type
                    break
            
            if valid_action is None:
                return 'Invalid', None
            
            # Special validation: Finish command must have content
            content = json_action.get('content', None)
            if valid_action == 'Finish' and (content is None or content == ""):
                return 'Invalid', None
            
            return valid_action, content
        except Exception as e:
            # Log the parsing error for debugging
            from loguru import logger
            logger.debug(f"JSON parsing error for action: '{action}', error: {e}")
            return 'Invalid', None
    else:
        pattern = r'^(\w+)\[(.*)\]$'
        match = re.match(pattern, action)

        if match:
            action_type = match.group(1)
            argument = match.group(2)
            return action_type, argument
        else:
            return 'Invalid', None

def parse_raw_answer(answer: str, *args, **kwargs) -> dict[str, bool | str]:
    return {
        'valid': True,
        'answer': answer
    }

def parse_rating_answer(answer: str | int | float, json_mode: bool = False, *args, **kwargs) -> dict[str, float | str]:
    try:
        answer = float(answer)
        if answer < 1 or answer > 5:
            return {
                'valid': False,
                'answer': 0,
                'message': 'Rating should be in range [1, 5].'
            }
    except (ValueError, TypeError):
        return {
            'valid': False,
            'answer': 0,
            'message': 'Rating should be a float number.'
        }
    except Exception:
        return {
            'valid': False,
            'answer': 0,
            'message': 'Other Exception when parsing rating.'
        }
    return {
        'valid': True,
        'answer': answer
    }

def parse_ranking_answer(answer: str | Any, gt_answer: int, n_candidate: int = None, json_mode: bool = False, *args, **kwargs) -> dict[str, bool | list[int]]:
    if not json_mode:
        candidates = answer.split(',')
    else:
        if isinstance(answer, list):
            candidates = answer
        elif isinstance(answer, str):
            # Try to parse as a literal list first (e.g., "[1, 2, 3]")
            if answer.strip().startswith('[') and answer.strip().endswith(']'):
                try:
                    import ast
                    parsed_list = ast.literal_eval(answer.strip())
                    if isinstance(parsed_list, list):
                        candidates = parsed_list
                    else:
                        candidates = answer.split(',')
                except (ValueError, SyntaxError):
                    candidates = answer.split(',')
            else:
                candidates = answer.split(',')
        else:
            return {
                'valid': False,
                'answer': [],
                'message': 'Answer should be a permutated list of candidate ids.'
            }
    
    try:
        length = len(candidates)
    except TypeError:
        return {
            'valid': False,
            'answer': [],
            'message': 'Answer should be a permutated list of candidate ids.'
        }
    except Exception:
        return {
            'valid': False,
            'answer': [],
            'message': 'Other Exception when parsing ranking answer.'
        }
    
    # If n_candidate is not provided, we can't validate the length, but we can still validate the format
    if n_candidate is not None and length != n_candidate:
        return {
            'valid': False,
            'answer': [],
            'message': f'Answer should contain only a list of {n_candidate} ids, which is the same as the number of candidates in the question.'
        }
    
    try:
        answer = [int(c) for c in candidates]
        # For ranking tasks, we don't need to validate that gt_answer is in the list
        # The agent is ranking the retrieved candidates, not including the ground truth
        # Just ensure all IDs are valid integers
        return {
            'valid': True,
            'answer': answer
        }
    except (ValueError, TypeError):
        return {
            'valid': False,
            'answer': [],
            'message': f'The ids in the answer list should be integers. Received: {answer}. Valid format: [1063, 151, 274, 225, 609, 25] (array of integers, NOT string)'
        }
    
    return {
        'valid': True,
        'answer': answer
    }

def parse_answer(type: str, *args, **kwargs) -> dict[str, Any]:
    """Parse answer.

    Args:
        `type` (`str`): Task type. Other arguments are passed to the corresponding parsing function.
    Raises:
        `NotImplementedError`: Unsupported task type.
    Returns:
        `dict[str, Any]`: Parsed answer, including `valid`, `answer`, and `message`. `valid` indicates whether the answer is valid. `answer` is the parsed answer. `message` is the error message if the answer is invalid (otherwise not included).
    """
    if type == 'qa' or type == 'chat' or type == 'gen':
        return parse_raw_answer(*args, **kwargs)
    elif type == 'rp':
        return parse_rating_answer(*args, **kwargs)
    elif type == 'sr' or type == 'rr':
        return parse_ranking_answer(*args, **kwargs)
    else:
        raise NotImplementedError(f'Unsupported task: {type}')

def init_answer(type: str) -> Any:
    """Initialize answer.

    Args:
        `type` (`str`): Task type.
    Raises:
        `NotImplementedError`: Unsupported task type.
    Returns:
        `Any`: Initialized answer. Different types of answers are returned for different tasks.
    """
    if type == 'qa' or type == 'chat' or type == 'gen':
        return ''
    elif type == 'rp':
        return 0
    elif type == 'sr' or type == 'rr':
        return []
    else:
        raise NotImplementedError(f'Unsupported task: {type}')
