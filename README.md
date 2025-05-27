# DiFFRL: Stock Trading & Robotics Simulation with PPO and AHAC

**DiFFRL** combines cutting-edge reinforcement learning (RL) for **algorithmic stock trading** and **robotics simulation** under a unified framework. We leverage historical market data from Dow Jones 30 (with plans for NASDAQ‑100 filtering) and sensor-driven robotic environments to develop, train, and evaluate RL agents using Proximal Policy Optimization (PPO) and a custom Adaptive Hierarchical Actor-Critic (AHAC) algorithm.

---

## Key Features

### Stock Trading
- **Data Aggregation**: Fetch 5‑minute interval data (∼4,000 rows) via Alpaca API or Yahoo Finance, forward‑fill missing values, and support multiple timeframes (5m, 30m, 1h, 2h, 4h) with a single parameter change.
- **RL Strategies**: Train and compare PPO, AHAC, SAC, SHAC, and optional SVG agents.
- **Real‑Time Execution**: Paper‑trade or live‑trade through Alpaca’s REST API.
- **Backtesting & Analysis**: Custom callbacks, reward‑curve plotting, and asset‑growth visualization (e.g., 8% Q3 increase).
- **Hyperparameter Tuning**: Integrated W&B sweeps for automated optimization.

### Robotics Simulation
- **Webots Integration**: Simulate robots performing navigation and obstacle avoidance.
- **Sensor Suites**: GPS, LiDAR, compass, gyro streams feed into custom environments.
- **Multi‑Agent Training**: Support for PPO and AHAC in Gym‑wrapped Webots environments.

### Infrastructure & Configurations
- **Hydra**: Modular configs for algorithms, environments, logging, and intervals.
- **Version Control**: Git for codebase management and reproducibility.

---

## Tech Stack
| Component            | Tool / Library                        |
|----------------------|---------------------------------------|
| Language             | Python                                |
| RL Algos             | PPO, AHAC, SAC, SHAC, SVG             |
| Data API             | Alpaca‑Py, YahooDownloader            |
| Config               | Hydra                                 |
| Experiment Logging   | Weights & Biases (wandb)              |
| Simulation & RL      | Webots, FinRL, RL‑Games, SB3, Gymnasium|
| Visualization        | Matplotlib                            |
| Deployment & Scripts | bash, Python scripts                  |

---

## Setup

1. **Clone**
   ```bash
git clone <repo-url>
cd <repo-name>
```
2. **Install Dependencies**
   ```bash
pip install -r requirements.txt
pip install alpaca‑py pywebots hydra‑core rl‑games stable‑baselines3 gymnasium wandb
```
3. **Configure Alpaca**
   - Edit `src/AHACEnvWrapper.py`:
     ```python
api_key = "YOUR_API_KEY"
api_secret = "YOUR_SECRET"
api_base_url = "https://paper-api.alpaca.markets/v2"
```
4. **Install & Configure Webots** (for robotics)
   - Download from https://cyberbotics.com/.
   - Ensure `pywebots` is installed.
5. **Build**
   ```bash
python setup.py install
```

---

## Usage

### Stock Trading
1. **Data Download**
   ```python
from src.AHACEnvWrapper import alpaca_downloader

df = alpaca_downloader(START_DATE, END_DATE, DOW_30_TICKER, interval="5m")
```
2. **Training**
   ```bash
python src/train.py env=StockTradingEnv alg=ahac
```
3. **Backtesting & Visualization**
   ```python
from src.AHACEnvWrapper import plot_actions
plot_actions("results/assets_plot_<episode>.png")
```
4. **Hyperparameter Sweeps**
   - Enable W&B in Hydra config: `cfg.general.run_wandb = true`

### Robotics Simulation
1. **Launch Webots** with the appropriate world file.
2. **Train**
   ```bash
python src/train.py env=WebotsGymEnv alg=ppo
```
3. **Alternate Scripts**
   - `run_model = 0`: train
   - `run_model = 1`: test
   - `run_model = 2`: manual debug with LiDAR

---

## Configuration
- **Algorithms**: `cfg/alg/ahac.yaml`, `cfg/alg/ppo.yaml`, …
- **Environments**: `cfg/env/StockTradingEnv.yaml`, `cfg/env/WebotsEnv.yaml`
- **Intervals**: Set `cfg.general.interval = [5m,30m,1h,2h,4h]`
- **State Vector**: 301‑dim for 5m data, scaled appropriately for other intervals.

---

## Results
- **Asset Growth**: `results/assets_plot_<episode>.png`
- **Reward Trends**: `results/reward_plot_<interval>.png`
- **Logs**: `results/actions.csv`
- **Models**: `src/models/model_<run>_<timestamp>.zip`

