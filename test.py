import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from train import VizDoomGym  # Import environment from train.py

def evaluate_and_run(model_path="best_model_final"):
    if not os.path.exists(f"{model_path}.zip"):
        print(f"Error: Could not find model file at {model_path}.zip")
        print("Please run train.py first to generate a trained model.")
        return

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    print("Setting up rendered VizDoom environment...")
    env = VizDoomGym(render=True)

    # Statistical Evaluation
    print("Evaluating mean reward across 10 episodes...")
    try:
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"\nMean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    except Exception as e:
        print("Evaluation aborted (possibly window closed manually).", e)

    # Visual Showcase
    print("\nStarting visual episodes for observation...")
    for episode in range(5): 
        obs, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), None)
        done = False
        total_reward = 0
        
        while not done: 
            action, _ = model.predict(obs)
            result = env.step(action)
            obs, reward, done = result[0], result[1], result[2]
            total_reward += reward
            # Delay to make the rendering human-viewable
            time.sleep(0.02)
            
        print(f'Total Reward for Episode {episode + 1} = {total_reward}')
        time.sleep(2)

    print("Closing Environment...")
    env.close()

if __name__ == "__main__":
    evaluate_and_run()
