"""
Reference Prey Agent - Trained with Stable-Baselines3 PPO
Public reference implementation used for evaluation.
"""

import numpy as np
from pathlib import Path


class StudentAgent:
    """
    Reference prey agent trained with PPO.
    Used as the public opponent for student predator evaluation.
    """
    
    def __init__(self):
        model_path = Path(__file__).parent / "prey_final_model.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"Reference prey model not found at {model_path}")
        
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(str(model_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load reference prey model: {e}")
    
    def get_action(self, observation, agent_id: str):
        action, _ = self.model.predict(observation, deterministic=True)
        if isinstance(action, np.ndarray):
            return int(np.argmax(action))
        return action
