import matplotlib.pyplot as plt
import numpy as np
from transformers import TrainerCallback
from pathlib import Path

class PlotMetricsCallback(TrainerCallback):
    def __init__(self, output_dir: str, smoothing_window: int = 10):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_losses = []
        self.eval_losses = []
        self.eval_accuracies = []
        self.train_steps = []
        self.eval_steps = []
        self.smoothing_window = smoothing_window

    def _smooth_data(self, data, window_size):
        """Apply moving average smoothing to reduce noise"""
        if len(data) < window_size:
            return data
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            smoothed.append(np.mean(data[start:end]))
        return smoothed

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}

        if "loss" in logs:
            self.train_losses.append(logs["loss"])
            self.train_steps.append(state.global_step)

        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
            self.eval_steps.append(state.global_step)

        # Hugging Face prefixes metric keys from compute_metrics with 'eval_'
        if "eval_accuracy" in logs:
            self.eval_accuracies.append(logs["eval_accuracy"])
            if len(self.eval_steps) < len(self.eval_accuracies):
                self.eval_steps.append(state.global_step)

        if len(self.train_steps) == 0:
            return

        # Build plot with smoothed data
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        # Smooth the training data
        if len(self.train_losses) > 1:
            smooth_train_losses = self._smooth_data(self.train_losses, self.smoothing_window)
            ax.plot(self.train_steps, smooth_train_losses, label="Training Loss (smoothed)", color="#2563eb", linewidth=2.5, alpha=0.8)
            # Also show original data as thin line for comparison
            ax.plot(self.train_steps, self.train_losses, color="#2563eb", linewidth=0.8, alpha=0.3, label="Training Loss (raw)")

        if len(self.eval_losses) > 0 and len(self.eval_steps) > 0:
            smooth_eval_losses = self._smooth_data(self.eval_losses, self.smoothing_window)
            ax.plot(self.eval_steps[:len(smooth_eval_losses)], smooth_eval_losses, label="Eval Loss (smoothed)", color="#ef4444", linewidth=2.5, alpha=0.8)
            # Show original eval data as thin line
            ax.plot(self.eval_steps[:len(self.eval_losses)], self.eval_losses, color="#ef4444", linewidth=0.8, alpha=0.3, label="Eval Loss (raw)")

        ax.set_xlabel("Steps", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.set_title("Training Progress", fontsize=14, fontweight='bold')

        # Secondary axis for accuracy
        if len(self.eval_accuracies) > 0 and len(self.eval_steps) > 0:
            ax2 = ax.twinx()
            smooth_accuracies = self._smooth_data(self.eval_accuracies, self.smoothing_window)
            ax2.plot(self.eval_steps[:len(smooth_accuracies)], smooth_accuracies, label="Eval Accuracy (smoothed)", color="#10b981", linewidth=2.5, alpha=0.8)
            # Show original accuracy as thin line
            ax2.plot(self.eval_steps[:len(self.eval_accuracies)], self.eval_accuracies, color="#10b981", linewidth=0.8, alpha=0.3, label="Eval Accuracy (raw)")
            ax2.set_ylabel("Accuracy", fontsize=12)

            # Build combined legend
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=10)
        else:
            ax.legend(loc="upper right", fontsize=10)

        plt.tight_layout()
        out_path = self.output_dir / "training_progress.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
