"""
Prompt Loader Module.

This module provides a central loader for all LLM prompts used in the application.
It loads prompts from structured YAML files located in `mcp_server/prompts/` and caches
them in memory so they are only parsed once at startup.

Usage:
    from shared.prompt_loader import render_prompt, load_prompt

    system_prompt, user_prompt = render_prompt('classify_intent', user_input='hello')
"""

import os
import yaml
from typing import Dict, Tuple, Any

class PromptNotFoundError(Exception):
    """Raised when a prompt file cannot be found for the given prompt_id."""
    pass

class PromptRenderError(Exception):
    """Raised when rendering a prompt fails due to missing variables."""
    pass

# Cache of loaded prompts
_PROMPT_CACHE: Dict[str, Dict[str, Any]] = {}

def get_prompts_dir() -> str:
    """Returns the absolute path to the prompts directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "mcp_server", "prompts")

def load_prompt(prompt_id: str) -> Dict[str, Any]:
    """
    Loads a prompt configuration from its YAML file.
    
    Args:
        prompt_id: The ID (filename without extension) of the prompt to load.
        
    Returns:
        The parsed YAML dictionary.
        
    Raises:
        PromptNotFoundError: If the prompt YAML file does not exist.
    """
    if prompt_id in _PROMPT_CACHE:
        return _PROMPT_CACHE[prompt_id]
        
    prompts_dir = get_prompts_dir()
    prompt_path = os.path.join(prompts_dir, f"{prompt_id}.yaml")
    
    if not os.path.exists(prompt_path):
        raise PromptNotFoundError(f"Prompt '{prompt_id}' not found at {prompt_path}")
        
    with open(prompt_path, "r", encoding="utf-8") as f:
        try:
            prompt_data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise PromptRenderError(f"Error parsing YAML for prompt '{prompt_id}': {exc}")
            
    _PROMPT_CACHE[prompt_id] = prompt_data
    return prompt_data

def render_prompt(prompt_id: str, **kwargs) -> Tuple[str, str]:
    """
    Loads and renders a prompt's system and user templates with the provided kwargs.
    
    Args:
        prompt_id: The ID of the prompt to render.
        **kwargs: Variables to inject into the user_template and system template (if needed).
        
    Returns:
        A tuple of (system_prompt, user_prompt) fully rendered.
        
    Raises:
        PromptRenderError: If a template is missing a required variable from kwargs.
    """
    prompt_data = load_prompt(prompt_id)
    
    system_template = prompt_data.get("system", "")
    user_template = prompt_data.get("user_template", "")
    
    try:
        system_rendered = system_template.format(**kwargs)
        user_rendered = user_template.format(**kwargs)
    except KeyError as exc:
        raise PromptRenderError(f"Missing required variable {exc} when rendering prompt '{prompt_id}'")
        
    return system_rendered, user_rendered
