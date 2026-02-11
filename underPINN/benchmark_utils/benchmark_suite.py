import time
import json
import os

class BenchmarkTracker:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.metrics = {}

    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        print("--- Benchmark Timer Started ---")

    def stop(self):
        """Stop the timer."""
        self.end_time = time.time()

    def log(self, key, value):
        """Log specific metrics like final loss, epochs, etc."""
        self.metrics[key] = value

    def save(self, case_name, framework, filename="benchmark_results.json"):
        """Save results to a shared JSON file."""
        if self.start_time is None or self.end_time is None:
            print("Error: Timer was not started or stopped.")
            return

        duration = self.end_time - self.start_time
        
        # Calculate time per epoch if 'epochs' was logged
        time_per_epoch = 0
        if 'epochs' in self.metrics and self.metrics['epochs'] > 0:
            time_per_epoch = duration / self.metrics['epochs']

        entry = {
            "case": case_name,          # e.g., "LDC", "Burgers"
            "framework": framework,     # e.g., "JAX", "PyTorch"
            "total_time_s": round(duration, 4),
            "time_per_epoch_s": round(time_per_epoch, 6),
            **self.metrics              # Merges final_loss, etc.
        }

        # Load existing data
        data = []
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                pass # File created but empty or corrupted

        # Append new entry
        data.append(entry)

        # Write back
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"--- Results saved to {filename} ---")