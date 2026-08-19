import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np

class NetworkSlicingEnv(gym.Env):
    def __init__(self, csv_file_path='data/final_hybrid_5g_traffic.csv'):
        super(NetworkSlicingEnv, self).__init__()
        
        # Load Dataset
        self.df = pd.read_csv(csv_file_path)
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        
        # Total Available System Capacity (e.g., 1000 Mbps)
        self.total_capacity = 1000.0
        
        # Action Space: Continuous allocation ratios for [eMBB, URLLC, mMTC]
        self.action_space = spaces.Box(
            low=0.05, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Observation Space: [eMBB_lambda, URLLC_lambda, mMTC_lambda, eMBB_queue, URLLC_queue, mMTC_queue]
        self.observation_space = spaces.Box(
            low=0, high=1e6, shape=(6,), dtype=np.float32
        )
        
        # Queuing States
        self.queue_lengths = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Service Capacities per Packet/Unit for each slice profile
        # eMBB = Large Packets, URLLC = High Priority/Fast Service, mMTC = Small Packets
        self.packet_sizes = np.array([1500.0, 200.0, 100.0], dtype=np.float32) # Bytes

    def _get_current_arrivals(self):
        row = self.df.iloc[self.current_step]
        return np.array([row['embb_lambda'], row['urllc_lambda'], row['mmtc_lambda']], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.queue_lengths = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        arrivals = self._get_current_arrivals()
        state = np.concatenate([arrivals, self.queue_lengths], axis=0)
        return state, {}

    def step(self, action):
        arrivals = self._get_current_arrivals()
        
        # 1. Normalize Actions to distribute Total Bandwidth
        allocated_bandwidth = (action / np.sum(action)) * self.total_capacity
        
        # 2. Calculate Service Rate (\mu) based on Allocated Bandwidth and Packet Size
        # \mu = Bandwidth / Packet_Size
        service_rates = (allocated_bandwidth * 1e6) / (self.packet_sizes * 8.0) # Packets per second
        
        # 3. Queuing Dynamics (M/M/C Approximation for 1-second interval)
        # Served packets in this 1-second step
        served = np.minimum(self.queue_lengths + arrivals, service_rates)
        
        # Update Queues (New Queue = Old Queue + Arrivals - Served)
        self.queue_lengths = np.maximum(0.0, self.queue_lengths + arrivals - served)
        
        # 4. Queuing Delay & SLA Violations (SLA Penalty calculation)
        # URLLC needs ultra-low queue buildup (high penalty), eMBB needs stability
        weights = np.array([1.0, 10.0, 0.1], dtype=np.float32) # URLLC has 10x penalty
        
        queue_penalty = np.sum(weights * self.queue_lengths)
        reward = float(-queue_penalty)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        if not done:
            next_arrivals = self._get_current_arrivals()
        else:
            next_arrivals = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
        next_state = np.concatenate([next_arrivals, self.queue_lengths], axis=0)
        return next_state, reward, done, False, {}