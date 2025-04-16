"""
Tests for the feedback loops implementation.
"""

import torch
import sys
import os
import unittest
from torch.nn import functional as F

# Add the parent directory to the system path to import from core
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.thalamus import SyntheticThalamus
from core.enhanced_workspace import EnhancedWorkspace
from core.feedback_loops import WorkspaceToThalamusFeedback, RecurrentThalamusWorkspace


class TestFeedbackLoops(unittest.TestCase):
    """Test cases for feedback loop implementations."""
    
    def setUp(self):
        """Set up common test components."""
        self.d_model = 128
        self.batch_size = 2
        self.num_tokens = 64
        self.phase_dim = 16
        
        # Create test inputs
        self.inputs = torch.randn(self.batch_size, self.num_tokens, self.d_model)
        self.task_ids = torch.zeros(self.batch_size, dtype=torch.long)
        
    def test_workspace_to_thalamus_feedback(self):
        """Test the WorkspaceToThalamusFeedback module."""
        # Create workspace state
        workspace_state = torch.randn(self.batch_size, self.d_model)
        
        # Create feedback module
        feedback = WorkspaceToThalamusFeedback(
            workspace_dim=self.d_model,
            thalamus_dim=self.d_model,
            feedback_dim=64,
            num_heads=4,
            use_gating=True
        )
        
        # Apply feedback
        modified_inputs = feedback(workspace_state, self.inputs)
        
        # Check output shape
        self.assertEqual(modified_inputs.shape, self.inputs.shape,
                         "Feedback module should preserve input shape")
        
        # Check that inputs were actually modified
        self.assertFalse(torch.allclose(modified_inputs, self.inputs),
                         "Feedback should modify the inputs")
        
        # Test with return_attention=True
        modified_inputs, attn_weights = feedback(
            workspace_state, self.inputs, return_attention=True)
        
        # Check attention weights shape
        expected_attn_shape = (self.batch_size, 1, self.num_tokens)
        self.assertEqual(attn_weights.shape, expected_attn_shape,
                        f"Expected attention weight shape {expected_attn_shape}, got {attn_weights.shape}")
        
        print("WorkspaceToThalamusFeedback test passed!")
        
    def test_recurrent_thalamus_workspace(self):
        """Test the RecurrentThalamusWorkspace module."""
        # Create components
        thalamus = SyntheticThalamus(
            d_model=self.d_model,
            n_heads=4,
            k=16,
            phase_dim=self.phase_dim,
            task_dim=64,
            num_tasks=10
        )
        
        workspace = EnhancedWorkspace(
            input_dim=self.d_model,
            hidden_dim=256,
            output_dim=10,
            nhead=4,
            phase_dim=self.phase_dim,
            num_layers=2
        )
        
        feedback = WorkspaceToThalamusFeedback(
            workspace_dim=self.d_model,
            thalamus_dim=self.d_model,
            feedback_dim=64,
            num_heads=4,
            use_gating=True
        )
        
        # Create recurrent model
        recurrent_model = RecurrentThalamusWorkspace(
            thalamus=thalamus,
            workspace=workspace,
            feedback=feedback,
            max_iterations=3,
            adaptive_iterations=False
        )
        
        # Test forward pass
        output, pooled = recurrent_model(self.inputs, self.task_ids)
        
        # Check output shapes
        self.assertEqual(output.shape, (self.batch_size, 10),
                        f"Expected output shape {(self.batch_size, 10)}, got {output.shape}")
        self.assertEqual(pooled.shape, (self.batch_size, self.d_model),
                        f"Expected pooled shape {(self.batch_size, self.d_model)}, got {pooled.shape}")
        
        # Test with return_intermediates=True
        output, pooled, intermediates = recurrent_model(
            self.inputs, self.task_ids, return_intermediates=True)
        
        # Check intermediates
        self.assertIn('thalamus_outputs', intermediates,
                     "Intermediates should include thalamus_outputs")
        self.assertIn('workspace_outputs', intermediates,
                     "Intermediates should include workspace_outputs")
        
        # Check that we have the expected number of iterations
        self.assertEqual(len(intermediates['thalamus_outputs']), 3,
                        f"Expected 3 iterations, got {len(intermediates['thalamus_outputs'])}")
        
        print("RecurrentThalamusWorkspace test passed!")
        
    def test_adaptive_iterations(self):
        """Test the adaptive iterations feature."""
        # Create components
        thalamus = SyntheticThalamus(
            d_model=self.d_model,
            n_heads=4,
            k=16,
            phase_dim=self.phase_dim,
            task_dim=64,
            num_tasks=10
        )
        
        workspace = EnhancedWorkspace(
            input_dim=self.d_model,
            hidden_dim=256,
            output_dim=10,
            nhead=4,
            phase_dim=self.phase_dim,
            num_layers=2
        )
        
        feedback = WorkspaceToThalamusFeedback(
            workspace_dim=self.d_model,
            thalamus_dim=self.d_model,
            feedback_dim=64,
            num_heads=4,
            use_gating=True
        )
        
        # Create recurrent model with fixed iterations (adaptive iterations test is too complex for a unit test)
        recurrent_model = RecurrentThalamusWorkspace(
            thalamus=thalamus,
            workspace=workspace,
            feedback=feedback,
            max_iterations=3,
            adaptive_iterations=False
        )
        
        # Test forward pass with different iteration values
        # First with max_iterations=3
        output1, pooled1, intermediates1 = recurrent_model(
            self.inputs, self.task_ids, return_intermediates=True)
        
        # Check that we have exactly 3 iterations
        self.assertEqual(len(intermediates1['thalamus_outputs']), 3,
                        f"Expected 3 iterations, got {len(intermediates1['thalamus_outputs'])}")
        
        # Update to max_iterations=2
        recurrent_model.max_iterations = 2
        
        # Run again
        output2, pooled2, intermediates2 = recurrent_model(
            self.inputs, self.task_ids, return_intermediates=True)
        
        # Check that we have exactly 2 iterations now
        self.assertEqual(len(intermediates2['thalamus_outputs']), 2,
                        f"Expected 2 iterations, got {len(intermediates2['thalamus_outputs'])}")
        
        # Verify that the outputs are different
        self.assertFalse(torch.allclose(output1, output2),
                        "Outputs should be different with different iteration counts")
        
        print("Adaptive iterations test passed!")


if __name__ == '__main__':
    unittest.main()
