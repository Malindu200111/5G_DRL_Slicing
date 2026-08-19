from custom_env.slicing_env import NetworkSlicingEnv

print("1. Initializing 5G Slicing Environment...")
env = NetworkSlicingEnv()

obs, _ = env.reset()

print("\n--- Simulation Started ---")
print("Initial State [eMBB_lambda, URLLC_lambda, mMTC_lambda, Queues...]:")
print(obs)

print("\nSimulating first 50 seconds with random RL actions:")
for step in range(1, 51):
    action = env.action_space.sample()  # Random capacity allocation
    next_obs, reward, done, _, _ = env.step(action)
    print(f"\n[Second {step}]")
    print(f"Allocated Action Ratio: {action}")
    
    # String එක raw string (r"...") කර ඇති නිසා \lambda warning එක නොපැමිණේ
    print(r"Arrivals (\lambda) -> eMBB: " + f"{next_obs[0]}, URLLC: {next_obs[1]}, mMTC: {next_obs[2]}")
    print(f"Queue Lengths      -> eMBB: {next_obs[3]:.1f}, URLLC: {next_obs[4]:.1f}, mMTC: {next_obs[5]:.1f}")
    print(f"Step Reward        : {reward:.2f}")

print("\nSUCCESS: Environment execution verified!")