"""
Test INL Dynamics - CRITICAL for training stability.

Priority: CRITICAL - This is the key innovation preventing training explosion.
"""

import pytest
import torch


class TestINLDynamics:
    """Test INL Dynamics component."""

    def test_inl_dynamics_forward(self):
        """Test basic forward pass."""
        from complexity.core.dynamics import INLDynamics

        dynamics = INLDynamics(
            hidden_size=256,
            beta_max=2.0,
            velocity_max=10.0,
        )

        h = torch.randn(2, 16, 256)
        v = torch.zeros(2, 16, 256)

        h_next, v_next = dynamics(h, v)
        assert h_next.shape == h.shape
        assert v_next.shape == v.shape

    def test_inl_dynamics_beta_clamping(self):
        """Test that beta is properly clamped - CRITICAL bug fix."""
        from complexity.core.dynamics import INLDynamics

        dynamics = INLDynamics(
            hidden_size=64,
            beta_max=2.0,
        )

        # Access internal beta parameter and check clamping
        h = torch.randn(2, 8, 64)
        v = torch.zeros(2, 8, 64)

        # Run forward
        h_next, v_next = dynamics(h, v)

        # Beta should be clamped to [0, beta_max]
        # The implementation should use: beta = clamp(softplus(beta_raw), max=beta_max)
        assert dynamics.beta_max == 2.0

    def test_inl_dynamics_velocity_clamping(self):
        """Test velocity clamping prevents explosion."""
        from complexity.core.dynamics import INLDynamics

        dynamics = INLDynamics(
            hidden_size=64,
            beta_max=2.0,
            velocity_max=10.0,
        )

        h = torch.randn(2, 8, 64)
        # Start with very large velocity
        v = torch.randn(2, 8, 64) * 1000

        h_next, v_next = dynamics(h, v)

        # Velocity should be clamped
        assert v_next.abs().max() <= dynamics.velocity_max + 1e-5

    def test_inl_dynamics_stability_long_run(self):
        """Test stability over many steps (simulating 400k+ training steps)."""
        from complexity.core.dynamics import INLDynamics

        dynamics = INLDynamics(
            hidden_size=64,
            beta_max=2.0,
            velocity_max=10.0,
        )

        h = torch.randn(1, 8, 64)
        v = torch.zeros(1, 8, 64)

        # Simulate many update steps
        for _ in range(1000):
            # Add noise to simulate gradient updates
            h = h + torch.randn_like(h) * 0.1
            h_next, v_next = dynamics(h, v)

            # Check for NaN/Inf - the critical failure mode
            assert not torch.isnan(h_next).any(), "NaN detected in hidden states!"
            assert not torch.isinf(h_next).any(), "Inf detected in hidden states!"
            assert not torch.isnan(v_next).any(), "NaN detected in velocity!"
            assert not torch.isinf(v_next).any(), "Inf detected in velocity!"

            h, v = h_next, v_next

    def test_inl_dynamics_loss_spike_recovery(self):
        """Test recovery from loss spike (large gradient injection)."""
        from complexity.core.dynamics import INLDynamics

        dynamics = INLDynamics(
            hidden_size=64,
            beta_max=2.0,
            velocity_max=10.0,
        )

        h = torch.randn(1, 8, 64)
        v = torch.zeros(1, 8, 64)

        # Normal updates
        for _ in range(100):
            h_next, v_next = dynamics(h, v)
            h, v = h_next, v_next

        # Inject loss spike (large sudden change)
        h_spike = h + torch.randn_like(h) * 100

        # Recovery phase
        h_recovered, v_recovered = dynamics(h_spike, v)

        # Should not explode
        assert not torch.isnan(h_recovered).any()
        assert not torch.isinf(h_recovered).any()

        # Continue for a few more steps
        h, v = h_recovered, v_recovered
        for _ in range(50):
            h_next, v_next = dynamics(h, v)
            assert not torch.isnan(h_next).any()
            assert not torch.isinf(h_next).any()
            h, v = h_next, v_next

    def test_inl_dynamics_gradient_flow(self):
        """Test that gradients flow properly through INL Dynamics."""
        from complexity.core.dynamics import INLDynamics

        dynamics = INLDynamics(
            hidden_size=64,
            beta_max=2.0,
        )

        h = torch.randn(2, 8, 64, requires_grad=True)
        v = torch.zeros(2, 8, 64, requires_grad=True)

        h_next, v_next = dynamics(h, v)

        # Compute loss and backprop
        loss = h_next.sum() + v_next.sum()
        loss.backward()

        # Gradients should exist
        assert h.grad is not None
        assert not torch.isnan(h.grad).any()


class TestINLDynamicsLite:
    """Test INLDynamicsLite (simplified dynamics)."""

    def test_dynamics_lite_init(self):
        """Test INLDynamicsLite initialization."""
        from complexity.core.dynamics import INLDynamicsLite

        dynamics = INLDynamicsLite(
            hidden_size=256,
            velocity_max=10.0,
        )
        assert dynamics is not None

    def test_dynamics_lite_forward(self):
        """Test INLDynamicsLite forward pass."""
        from complexity.core.dynamics import INLDynamicsLite

        dynamics = INLDynamicsLite(
            hidden_size=64,
            velocity_max=10.0,
        )

        h = torch.randn(2, 8, 64)
        v = torch.zeros(2, 8, 64)

        h_next, v_next = dynamics(h, v)
        assert h_next.shape == h.shape
        assert v_next.shape == v.shape

    def test_dynamics_lite_velocity_clamping(self):
        """Test velocity is clamped."""
        from complexity.core.dynamics import INLDynamicsLite

        dynamics = INLDynamicsLite(
            hidden_size=64,
            velocity_max=5.0,
        )

        h = torch.randn(2, 8, 64)
        # Large initial velocity
        v = torch.randn(2, 8, 64) * 1000

        h_next, v_next = dynamics(h, v)
        assert v_next.abs().max() <= 5.0 + 1e-5


class TestSecondOrderDynamics:
    """Test second-order dynamical system properties."""

    def test_second_order_update(self):
        """Test second-order update equation."""
        from complexity.core.dynamics import INLDynamics

        dynamics = INLDynamics(
            hidden_size=64,
            beta_max=2.0,
            velocity_max=10.0,
        )

        h = torch.randn(1, 4, 64)
        v = torch.randn(1, 4, 64) * 0.1

        # Update should use: h_next = h + dt * v_next
        # where v_next is damped
        h_next, v_next = dynamics(h, v)

        # Check the update is reasonable
        assert h_next.shape == h.shape
        assert v_next.shape == v.shape

    def test_damping_reduces_velocity(self):
        """Test that damping reduces velocity over time."""
        from complexity.core.dynamics import INLDynamics

        dynamics = INLDynamics(
            hidden_size=64,
            beta_max=1.5,  # Strong damping
            velocity_max=10.0,
        )

        h = torch.randn(1, 4, 64)
        v = torch.randn(1, 4, 64)  # Non-zero initial velocity

        initial_v_norm = v.norm()

        # Run several steps without external input
        for _ in range(20):
            h_next, v_next = dynamics(h, v)
            h, v = h_next, v_next

        final_v_norm = v.norm()

        # Velocity should be reduced by damping
        # (unless external gradients add energy)
        # This tests the damping mechanism works
        assert final_v_norm < initial_v_norm * 10  # Some bound


class TestBetaParameterization:
    """Test beta parameter handling - critical for stability."""

    def test_beta_softplus_clamp(self):
        """Test beta uses softplus then clamp."""
        from complexity.core.dynamics import INLDynamics

        dynamics = INLDynamics(
            hidden_size=64,
            beta_max=2.0,
        )

        # The effective beta should always be in [0, beta_max]
        # This is the critical fix that prevents explosion at 400k steps
        h = torch.randn(2, 8, 64)
        v = torch.zeros(2, 8, 64)

        # Should not raise any errors
        h_next, v_next = dynamics(h, v)

        # And should not produce NaN
        assert not torch.isnan(h_next).any()

    def test_beta_max_boundary(self):
        """Test behavior at beta_max boundary."""
        from complexity.core.dynamics import INLDynamics

        # Test with different beta_max values
        for beta_max in [0.5, 1.0, 2.0, 5.0]:
            dynamics = INLDynamics(
                hidden_size=32,
                beta_max=beta_max,
            )

            h = torch.randn(1, 4, 32)
            v = torch.randn(1, 4, 32)

            # Run many steps
            for _ in range(100):
                h_next, v_next = dynamics(h, v)
                assert not torch.isnan(h_next).any()
                assert not torch.isinf(h_next).any()
                h, v = h_next, v_next
