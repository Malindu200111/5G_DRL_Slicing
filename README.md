# 🚀 Dynamic 5G Network Slicing via DRL & Queuing Theory (HIL-Enabled)

An advanced 5G Multi-Slice Dynamic Resource Allocation framework that bridges **Analytical Queuing Models ($M/M/c/K$)** with **Deep Reinforcement Learning (DQN / PPO)** and **Hardware-in-the-Loop (HIL)** emulation.

---

## 📌 Project Overview

Traditional 5G resource allocation algorithms struggle with highly dynamic, bursty traffic without risking SLA violations or resource under-utilization. Furthermore, standard model-free DRL approaches often treat network buffers as "black boxes," leading to slow convergence and unstable resource allocation.

This project proposes a novel **hybrid framework** that:
* **Models 5G Network Slices:** Simulates eMBB, URLLC, and mMTC slices using priority-based, finite-capacity $M/M/c/K$ queuing systems in **SimPy**.
* **DRL-Driven Optimization:** Trains DRL agents (DQN & PPO) to dynamically allocate bandwidth and scale processing servers ($c$).
* **Mathematically Guided Rewards:** Integrates **Little’s Law ($L_q = \lambda_a W_q$)** into the reinforcement reward engine to penalize packet drops and delay explicitly based on queuing dynamics.
* **Hardware-in-the-Loop (HIL):** Integrates ESP32 microcontrollers to inject real-world telemetry and packet streams via Socket Programming (UDP/IP).

---

## 🛠️ Tech Stack & Tools

* **Programming Language:** Python 3.x
* **Simulation Framework:** SimPy (Discrete Event Simulation)
* **Reinforcement Learning:** PyTorch / TensorFlow, Stable-Baselines3 (DQN, PPO)
* **Hardware:** ESP32 Microcontrollers (Wi-Fi UDP Traffic Generation)
* **Mathematical Framework:** Queuing Theory ($M/M/c/K$), Little's Law
* **Data Processing & Viz:** NumPy, Pandas, Matplotlib
