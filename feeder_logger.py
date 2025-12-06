"""Logger for Bottom Feeder training statistics."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir="training_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_log_{timestamp}.json"
        
        self.history = {
            # Basic metrics
            "epoch": [],
            "avg_fitness": [],
            "max_fitness": [],
            "min_fitness": [],
            "median_fitness": [],
            "fitness_std": [],
            "population": [],
            
            # Food consumption
            "total_food_eaten": [],
            
            # Behavioral metrics
            "avg_speed": [],
            "avg_hunger": [],
            "avg_energy": [],
            "max_energy": [],
            
            # Neural network metrics
            "weight_mean": [],
            "weight_std": [],
            "weight_change": [],
            "network_diversity": [],
            
            # Performance metrics
            "top_10_percent_fitness": [],
            "bottom_10_percent_fitness": [],
            "fitness_improvement": [],
        }
        
    def log(self, epoch, fish_stats):
        """
        Log comprehensive statistics for an epoch.
        
        fish_stats should be a dict containing:
        - alive_fish: list of alive fish
        - dead_fish: list of dead fish
        - all_fish: list of all fish (alive + dead)
        - prev_avg_fitness: previous epoch's average fitness (for improvement calc)
        - prev_weights: previous epoch's best weights (for change calc)
        """
        alive = fish_stats['alive_fish']
        dead = fish_stats['dead_fish']
        all_fish = fish_stats['all_fish']
        
        # Basic fitness metrics
        if alive:
            fitnesses = [f.fitness for f in alive]
            self.history["avg_fitness"].append(float(np.mean(fitnesses)))
            self.history["max_fitness"].append(float(np.max(fitnesses)))
            self.history["min_fitness"].append(float(np.min(fitnesses)))
            self.history["median_fitness"].append(float(np.median(fitnesses)))
            self.history["fitness_std"].append(float(np.std(fitnesses)))
            
            # Top/Bottom percentiles
            sorted_fit = sorted(fitnesses, reverse=True)
            top_10_idx = max(1, len(sorted_fit) // 10)
            bottom_10_idx = max(1, len(sorted_fit) // 10)
            self.history["top_10_percent_fitness"].append(float(np.mean(sorted_fit[:top_10_idx])))
            self.history["bottom_10_percent_fitness"].append(float(np.mean(sorted_fit[-bottom_10_idx:])))
        else:
            # Extinction
            self.history["avg_fitness"].append(0.0)
            self.history["max_fitness"].append(0.0)
            self.history["min_fitness"].append(0.0)
            self.history["median_fitness"].append(0.0)
            self.history["fitness_std"].append(0.0)
            self.history["top_10_percent_fitness"].append(0.0)
            self.history["bottom_10_percent_fitness"].append(0.0)
        
        # Population metrics
        alive_pop = len(alive)
        self.history["population"].append(alive_pop)
        
        # Total food eaten (sum of all fish fitness / 10, since fitness += count * 10)
        # Fitness = survival_time + food_eaten * 10
        # Approximate food eaten from fitness
        if alive:
            total_food = sum((f.fitness - 30.0) / 10.0 for f in alive)  # Subtract 30s survival time
            self.history["total_food_eaten"].append(float(max(0, total_food)))
        else:
            self.history["total_food_eaten"].append(0.0)
        
        # Behavioral metrics
        if alive:
            self.history["avg_speed"].append(float(np.mean([f.speed for f in alive])))
            self.history["avg_hunger"].append(float(np.mean([f.hunger for f in alive])))
            self.history["avg_energy"].append(float(np.mean([f.energy for f in alive])))
            self.history["max_energy"].append(float(np.max([f.energy for f in alive])))
        else:
            self.history["avg_speed"].append(0.0)
            self.history["avg_hunger"].append(0.0)
            self.history["avg_energy"].append(0.0)
            self.history["max_energy"].append(0.0)
        
        # Neural network metrics
        if alive:
            best_fish = max(alive, key=lambda f: f.fitness)
            weights = best_fish.net.get_all_weights()
            self.history["weight_mean"].append(float(np.mean(weights)))
            self.history["weight_std"].append(float(np.std(weights)))
            
            # Weight change from previous epoch
            if fish_stats.get('prev_weights') is not None:
                prev_weights = fish_stats['prev_weights']
                weight_diff = np.abs(weights - prev_weights)
                self.history["weight_change"].append(float(np.mean(weight_diff)))
            else:
                self.history["weight_change"].append(0.0)
            
            # Network diversity (std of all fish weights)
            all_weights = [f.net.get_all_weights() for f in alive[:min(20, len(alive))]]  # Sample 20
            if len(all_weights) > 1:
                diversity = np.mean([np.std([w[i] for w in all_weights]) for i in range(len(all_weights[0]))])
                self.history["network_diversity"].append(float(diversity))
            else:
                self.history["network_diversity"].append(0.0)
        else:
            self.history["weight_mean"].append(0.0)
            self.history["weight_std"].append(0.0)
            self.history["weight_change"].append(0.0)
            self.history["network_diversity"].append(0.0)
        
        # Fitness improvement
        if fish_stats.get('prev_avg_fitness') is not None and self.history["avg_fitness"][-1] > 0:
            improvement = self.history["avg_fitness"][-1] - fish_stats['prev_avg_fitness']
            self.history["fitness_improvement"].append(float(improvement))
        else:
            self.history["fitness_improvement"].append(0.0)
        
        self.history["epoch"].append(epoch)
        
        # Save to file
        with open(self.log_file, "w") as f:
            json.dump(self.history, f, indent=2)
            
    def plot(self):
        """Generate comprehensive multi-panel training visualization"""
        epochs = self.history["epoch"]
        if not epochs:
            return
        
        # Create figure with 2x4 subplots
        fig = plt.figure(figsize=(20, 10))
        
        # 1. Fitness Evolution (with median)
        ax1 = plt.subplot(2, 4, 1)
        ax1.plot(epochs, self.history["avg_fitness"], label="Mean", linewidth=2.5, color='#2ecc71')
        ax1.plot(epochs, self.history["median_fitness"], label="Median", linewidth=2.5, color='#f39c12', linestyle='--')
        ax1.plot(epochs, self.history["max_fitness"], label="Max", linewidth=2, color='#3498db')
        ax1.plot(epochs, self.history["min_fitness"], label="Min", alpha=0.6, color='#e74c3c')
        ax1.fill_between(epochs, 
                         np.array(self.history["avg_fitness"]) - np.array(self.history["fitness_std"]),
                         np.array(self.history["avg_fitness"]) + np.array(self.history["fitness_std"]),
                         alpha=0.2, color='#2ecc71', label="±1 Std")
        ax1.set_xlabel("Epoch", fontsize=10)
        ax1.set_ylabel("Fitness (Food Eaten)", fontsize=10)
        ax1.set_title("Fitness Evolution", fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. Total Food Eaten
        ax2 = plt.subplot(2, 4, 2)
        ax2.plot(epochs, self.history["total_food_eaten"], color="#27ae60", linewidth=2.5)
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("Total Food Eaten", color="#27ae60", fontsize=10)
        ax2.set_title("Population Food Consumption", fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor="#27ae60")
        
        # 3. Fitness Distribution
        ax3 = plt.subplot(2, 4, 3)
        ax3.plot(epochs, self.history["top_10_percent_fitness"], label="Top 10%", linewidth=2.5, color='#9b59b6')
        ax3.plot(epochs, self.history["avg_fitness"], label="Average", linewidth=2, color='#34495e')
        ax3.plot(epochs, self.history["bottom_10_percent_fitness"], label="Bottom 10%", linewidth=2, color='#95a5a6')
        ax3.set_xlabel("Epoch", fontsize=10)
        ax3.set_ylabel("Fitness", fontsize=10)
        ax3.set_title("Fitness Distribution", fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 4. Population Size
        ax4 = plt.subplot(2, 4, 4)
        ax4.plot(epochs, self.history["population"], linewidth=2.5, color='#16a085')
        ax4.axhline(y=150, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label="Target (150)")
        ax4.set_xlabel("Epoch", fontsize=10)
        ax4.set_ylabel("Alive Fish", fontsize=10)
        ax4.set_title("Population Size", fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # 5. Behavioral Metrics
        ax5 = plt.subplot(2, 4, 5)
        ax5_twin = ax5.twinx()
        ax5.plot(epochs, self.history["avg_speed"], color="#3498db", linewidth=2.5, label="Speed")
        ax5_twin.plot(epochs, self.history["avg_hunger"], color="#e74c3c", linewidth=2.5, label="Hunger")
        ax5.set_xlabel("Epoch", fontsize=10)
        ax5.set_ylabel("Avg Speed", color="#3498db", fontsize=10)
        ax5_twin.set_ylabel("Avg Hunger", color="#e74c3c", fontsize=10)
        ax5.set_title("Behavior: Speed vs Hunger", fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='y', labelcolor="#3498db")
        ax5_twin.tick_params(axis='y', labelcolor="#e74c3c")
        
        # 6. Energy Levels
        ax6 = plt.subplot(2, 4, 6)
        ax6.plot(epochs, self.history["avg_energy"], label="Avg Energy", linewidth=2.5, color='#f39c12')
        ax6.plot(epochs, self.history["max_energy"], label="Max Energy", linewidth=2, alpha=0.7, color='#d35400')
        ax6.axhline(y=40, color='#27ae60', linestyle='--', linewidth=2, alpha=0.6, label="Repro Threshold")
        ax6.set_xlabel("Epoch", fontsize=10)
        ax6.set_ylabel("Energy", fontsize=10)
        ax6.set_title("Energy Levels", fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        # 7. Neural Network Weights
        ax7 = plt.subplot(2, 4, 7)
        ax7.plot(epochs, self.history["weight_mean"], label="Mean Weight", linewidth=2.5, color='#8e44ad')
        ax7.fill_between(epochs,
                         np.array(self.history["weight_mean"]) - np.array(self.history["weight_std"]),
                         np.array(self.history["weight_mean"]) + np.array(self.history["weight_std"]),
                         alpha=0.3, color='#8e44ad', label="±1 Std")
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax7.set_xlabel("Epoch", fontsize=10)
        ax7.set_ylabel("Weight Value", fontsize=10)
        ax7.set_title("Neural Network Weights", fontsize=12, fontweight='bold')
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3)
        
        # 8. Evolution Dynamics
        ax8 = plt.subplot(2, 4, 8)
        ax8_twin = ax8.twinx()
        ax8.plot(epochs, self.history["weight_change"], color="#9b59b6", linewidth=2.5, label="Weight Change")
        ax8_twin.plot(epochs, self.history["network_diversity"], color="#c0392b", linewidth=2.5, label="Diversity")
        ax8.set_xlabel("Epoch", fontsize=10)
        ax8.set_ylabel("Avg Weight Change", color="#9b59b6", fontsize=10)
        ax8_twin.set_ylabel("Network Diversity", color="#c0392b", fontsize=10)
        ax8.set_title("Evolution Dynamics", fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        ax8.tick_params(axis='y', labelcolor="#9b59b6")
        ax8_twin.tick_params(axis='y', labelcolor="#c0392b")
        
        plt.suptitle(f"Training Analysis - {len(epochs)} Epochs", fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.log_file.with_suffix(".png"), dpi=150, bbox_inches='tight')
        plt.close()
