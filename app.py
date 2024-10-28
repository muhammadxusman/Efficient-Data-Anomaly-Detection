import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import time


class DataStreamSimulator:
    """Simulates a real-time data stream with seasonal trends and occasional anomalies"""
    
    def __init__(self, seasonal_pattern='daily', noise_level=0.1, trend_rate=0.001):
        self.seasonal_pattern = seasonal_pattern
        self.noise_level = noise_level
        self.trend_rate = trend_rate
        self.time_index = 0
        
    def _generate_baseline(self):
        """Generate baseline value based on seasonal pattern and trend"""
        hour = self.time_index % 24 if self.seasonal_pattern == 'daily' else 0
        baseline = 10 + 5 * np.sin(2 * np.pi * hour / 24)
        trend = self.trend_rate * self.time_index  # Incremental trend over time
        return baseline + trend
    
    def get_next_value(self):
        """Generate the next value in the stream, with occasional anomalies"""
        baseline = self._generate_baseline()
        noise = np.random.normal(0, self.noise_level)
        
        # Inject occasional anomalies (5% chance for better observability)
        if np.random.random() < 0.05:
            anomaly = baseline * np.random.choice([3, 0.1, 1 + np.random.normal(0, 1)])
            self.time_index += 1
            return anomaly
        
        self.time_index += 1
        return baseline + noise


class AdaptiveAnomalyDetector:
    """Detects anomalies in a real-time data stream with adaptive thresholds"""
    
    def __init__(self, window_size=50, sensitivity=1.5):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.values = deque(maxlen=window_size)
    
    def update_window(self, value):
        """Add new value to the window and calculate mean and deviation dynamically"""
        self.values.append(value)
        avg = np.mean(self.values)
        std_dev = np.std(self.values)
        return avg, std_dev

    def is_anomaly(self, value):
        """Determine if a value is anomalous based on adaptive threshold"""
        if len(self.values) < self.window_size:
            self.values.append(value)
            return False
        
        avg, std_dev = self.update_window(value)
        threshold = self.sensitivity * std_dev
        return abs(value - avg) > threshold


class StreamVisualizer:
    """Real-time visualization of data stream and anomalies"""
    
    def __init__(self, history_size=1000):
        self.history_size = history_size
        self.timestamps = deque(maxlen=history_size)
        self.values = deque(maxlen=history_size)
        self.anomalies = deque(maxlen=history_size)
        
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line, = self.ax.plot([], [], 'b-', label='Data Stream')
        self.anomaly_scatter, = self.ax.plot([], [], 'ro', label='Anomalies')
        self.ax.legend()
    
    def update(self, timestamp, value, is_anomaly):
        """Update visualization with new data point"""
        self.timestamps.append(timestamp)
        self.values.append(value)
        
        if is_anomaly:
            self.anomalies.append((timestamp, value))
        
        self.line.set_data(list(self.timestamps), list(self.values))
        
        if self.anomalies:
            x, y = zip(*self.anomalies)
            self.anomaly_scatter.set_data(x, y)
        
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main():
    """Main function to run the adaptive anomaly detection and visualization"""
    
    # Initialize components
    simulator = DataStreamSimulator(seasonal_pattern='daily', noise_level=0.2, trend_rate=0.002)
    detector = AdaptiveAnomalyDetector(window_size=50, sensitivity=1.8)
    visualizer = StreamVisualizer(history_size=500)
    
    try:
        while True:
            value = simulator.get_next_value()
            timestamp = datetime.now()
            is_anomaly = detector.is_anomaly(value)
            visualizer.update(timestamp, value, is_anomaly)
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Stopping data stream...")
        plt.close()

if __name__ == "__main__":
    main()
