"""Logger for Bottom Feeder training statistics."""
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir="training_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_log_{timestamp}.json"
        
        self.history = {
            "epoch": [],
            "avg_fitness": [],
            "max_fitness": [],
            "population": []
        }
        
    def log(self, epoch, avg_fitness, max_fitness, population):
        self.history["epoch"].append(epoch)
        self.history["avg_fitness"].append(avg_fitness)
        self.history["max_fitness"].append(max_fitness)
        self.history["population"].append(population)
        
        # Save to file
        with open(self.log_file, "w") as f:
            json.dump(self.history, f, indent=2)
            
    def plot(self):
        epochs = self.history["epoch"]
        if not epochs:
            return
            
        plt.figure(figsize=(12, 5))
        
        # Fitness Plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["avg_fitness"], label="Avg Fitness")
        plt.plot(epochs, self.history["max_fitness"], label="Max Fitness")
        plt.xlabel("Epoch")
        plt.ylabel("Fitness")
        plt.title("Fitness over Time")
        plt.legend()
        plt.grid(True)
        
        # Population Plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history["population"], color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Population")
        plt.title("Population Size")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_file.with_suffix(".png"))
        plt.close()
