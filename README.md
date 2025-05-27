
# Stock Trading & Robotics Simulation with DeepLearning

**Stock Trading & Robotics Simulation with DeepLearning** is a unified framework for **reinforcement learning** in both **algorithmic stock trading** and **robotics simulation**. It combines historical market data (e.g., Dow Jones 30) and sensor-based robotic environments to train and evaluate agents using **Proximal Policy Optimization (PPO)** and a custom **Adaptive Hierarchical Actor-Critic (AHAC)** algorithm.

---

## Key Features

### Stock Trading
- **Data Aggregation**  
  Retrieve 5-minute interval data (~4,000 rows) using **Alpaca API** or **Yahoo Finance**, with automatic forward-filling and multi-timeframe support (5m, 30m, 1h, 2h, 4h).
  
- **RL Strategies**  
  Train and compare **PPO**, **AHAC**, **SAC**, **SHAC**, and optionally **SVG** agents.

- **Real-Time Execution**  
  Use **Alpaca REST API** for live or paper trading.

- **Backtesting & Analysis**  
  Custom callbacks, reward plots, and asset growth visualizations (e.g., 8% Q3 increase).

- **Hyperparameter Tuning**  
  Automated sweeps via **Weights & Biases (wandb)**.

### Robotics Simulation
- **Webots Integration**  
  Simulate robotic tasks such as navigation and obstacle avoidance.

- **Sensor Input**  
  Use GPS, LiDAR, compass, and gyroscope data streams in custom Gym environments.

- **Multi-Agent Training**  
  Train agents using PPO and AHAC in Webots-Gym wrapped environments.

### Infrastructure & Configuration
- **Hydra**  
  Modular YAML configuration for algorithms, environments, logging, and intervals.

- **Version Control**  
  Git-based project organization for reproducibility.

---

## Tech Stack

| Component            | Tool / Library                         |
|---------------------|----------------------------------------|
| Language             | Python                                 |
| RL Algorithms        | PPO, AHAC, SAC, SHAC, SVG              |
| Data API             | Alpaca‑Py, YahooDownloader             |
| Configuration        | Hydra                                  |
| Experiment Logging   | Weights & Biases (wandb)               |
| Simulation & RL      | Webots, FinRL, RL‑Games, SB3, Gymnasium|
| Visualization        | Matplotlib                             |
| Scripts              | Python, bash                           |

---

## Setup

1. **Clone the Repo**
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install alpaca‑py pywebots hydra‑core rl‑games stable‑baselines3 gymnasium wandb
   ```

3. **Configure Alpaca API**
   In `src/AHACEnvWrapper.py`, update:
   ```python
   api_key = "YOUR_API_KEY"
   api_secret = "YOUR_SECRET"
   api_base_url = "https://paper-api.alpaca.markets/v2"
   ```

4. **Install & Configure Webots (for Robotics)**
   - Download from: https://cyberbotics.com/
   - Install `pywebots` Python bindings.

5. **Build the Package**
   ```bash
   python setup.py install
   ```

---

## Usage

### Stock Trading

1. **Download Data**
   ```python
   from src.AHACEnvWrapper import alpaca_downloader

   df = alpaca_downloader(START_DATE, END_DATE, DOW_30_TICKER, interval="5m")
   ```

2. **Train Agent**
   ```bash
   python src/train.py env=StockTradingEnv alg=ahac
   ```

3. **Visualize Backtest**
   ```python
   from src.AHACEnvWrapper import plot_actions

   plot_actions("results/assets_plot_<episode>.png")
   ```

4. **Hyperparameter Sweeps (W&B)**
   Enable in Hydra config:
   ```yaml
   cfg.general.run_wandb = true
   ```

### Robotics Simulation

1. **Launch Webots**
   Open the appropriate `.wbt` world file.

2. **Train Agent**
   ```bash
   python src/train.py env=WebotsGymEnv alg=ppo
   ```

3. **Optional Modes**
   - `run_model = 0`: Train
   - `run_model = 1`: Test
   - `run_model = 2`: Debug mode (manual LiDAR streaming)

---

## Configuration Files

- **Algorithms**  
  `cfg/alg/ahac.yaml`, `cfg/alg/ppo.yaml`, ...

- **Environments**  
  `cfg/env/StockTradingEnv.yaml`, `cfg/env/WebotsEnv.yaml`

- **Intervals**  
  Set via: `cfg.general.interval = [5m,30m,1h,2h,4h]`

- **State Vector**  
  301-dim for 5m data, automatically scaled for others.

---

## Results

| Output            | Location                                |
|------------------|------------------------------------------|
| Asset Growth      | `results/assets_plot_<episode>.png`     |
| Reward Curve      | `results/reward_plot_<interval>.png`    |
| Action Logs       | `results/actions.csv`                   |
| Trained Models    | `src/models/model_<run>_<timestamp>.zip`|
