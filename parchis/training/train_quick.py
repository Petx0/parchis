"""
Quick training script for testing and development.

This is a simplified version for quick experiments with shorter training times.
Good for testing changes and iterating on the environment.
"""

from parchis.training.train_ppo import train

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Quick Training Mode - Short run for testing")
    print("=" * 70)
    print("\nThis will train for 10,000 timesteps (~1-2 minutes)")
    print("Good for testing and quick iterations")
    print("\nFor full training, use: python -m parchis.training.train_ppo --timesteps 1000000")
    print("=" * 70 + "\n")

    # Train with smaller timesteps for quick testing
    model = train(
        total_timesteps=10_000,      # Quick training
        num_players=4,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        checkpoint_freq=10_000,       # More frequent checkpoints
        eval_freq=None,               # Disabled by default (evaluate at end only)
        n_eval_episodes=5,            # Fewer evaluation episodes
        save_path="./models",
        log_path="./logs",
        model_name="parchis_quick_test",
        verbose=1
    )

    print("\n" + "=" * 70)
    print("Quick training completed!")
    print("=" * 70)
    print("\nModel saved to: ./models/parchis_quick_test/")
    print("\nTo continue training from this checkpoint:")
    print("  python -m parchis.training.train_continue \\")
    print("      --model-path ./models/parchis_quick_test/final_model \\")
    print("      --timesteps 1000000")
    print("\nTo evaluate:")
    print("  python -m parchis.training.train_ppo --evaluate ./models/parchis_quick_test/final_model")
    print("=" * 70)
