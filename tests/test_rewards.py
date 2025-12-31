import unittest
import torch
from rewards.composite import CompositeReward

class TestRewardStructure(unittest.TestCase):
    """Basic tests to ensure reward modules load correctly."""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_composite_initialization(self):
        """Test if CompositeReward can be initialized without error."""
        try:
            reward = CompositeReward(device=self.device)
            self.assertIsNotNone(reward)
        except Exception as e:
            self.fail(f"CompositeReward initialization failed: {e}")

if __name__ == '__main__':
    unittest.main()
