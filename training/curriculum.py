import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd
from trading_env.institutional_perp_env import InstitutionalPerpetualEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import wandb
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import os

from training.loggers import WandBLogger

logger = logging.getLogger(__name__)

@dataclass
class CurriculumStage:
    name: str
    description: str
    difficulty: float
    env_params: Dict
    required_metrics: Dict[str, float]
    duration: int

class CurriculumManager:
    def __init__(
        self,
        stages: List[CurriculumStage],
        evaluation_window: int = 100,
        progression_threshold: float = 0.8
    ):
        self.stages = sorted(stages, key=lambda x: x.difficulty)
        self.current_stage_idx = 0
        self.evaluation_window = evaluation_window
        self.progression_threshold = progression_threshold
        
        # Performance tracking
        self.stage_metrics = []
        self.current_metrics = {
            'returns': [],
            'sharpe': [],
            'max_drawdown': [],
            'win_rate': [],
            'success_rate': []
        }
        
    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage"""
        return self.stages[self.current_stage_idx]
        
    def update_metrics(self, metrics: Dict):
        """Update performance metrics"""
        for key in self.current_metrics:
            if key in metrics:
                self.current_metrics[key].append(metrics[key])
                
        # Keep only recent metrics
        for key in self.current_metrics:
            self.current_metrics[key] = self.current_metrics[key][-self.evaluation_window:]
            
    def should_progress(self) -> bool:
        """Check if agent should progress to next stage"""
        if len(self.current_metrics['returns']) < self.evaluation_window:
            return False
            
        current_stage = self.get_current_stage()
        
        # Check if required metrics are met
        for metric, required_value in current_stage.required_metrics.items():
            if metric not in self.current_metrics:
                continue
                
            current_value = np.mean(self.current_metrics[metric])
            if current_value < required_value:
                return False
                
        # Check success rate
        success_rate = np.mean(self.current_metrics['success_rate'])
        return success_rate >= self.progression_threshold
        
    def progress(self) -> bool:
        """Progress to next stage if available"""
        if self.current_stage_idx >= len(self.stages) - 1:
            return False
            
        # Store stage performance
        self.stage_metrics.append({
            'stage': self.get_current_stage().name,
            'metrics': {
                key: np.mean(values)
                for key, values in self.current_metrics.items()
            }
        })
        
        # Move to next stage
        self.current_stage_idx += 1
        self.current_metrics = {key: [] for key in self.current_metrics}
        
        logger.info(f"Progressing to stage: {self.get_current_stage().name}")
        return True
        
    def get_stage_history(self) -> List[Dict]:
        """Get history of stage performances"""
        return self.stage_metrics

class CurriculumScheduler:
    """Manages progressive difficulty scaling during training"""
    def __init__(
        self,
        initial_balance: float = 1e6,
        max_leverage: float = 20,
        n_envs: int = 8,
        data_start_date: datetime = datetime(2020, 1, 1),
        data_end_date: datetime = datetime(2023, 12, 31)
    ):
        self.initial_balance = initial_balance
        self.max_leverage = max_leverage
        self.n_envs = n_envs
        self.data_start_date = data_start_date
        self.data_end_date = data_end_date
        
        # Curriculum stages
        self.stages = [
            {
                'name': 'basic_trading',
                'description': 'Learn basic trading with low leverage',
                'difficulty': 0.2,
                'max_leverage': 2,
                'assets': ['BTC/USD:USD'],
                'window_size': 50,
                'timesteps': 1_000_000
            },
            {
                'name': 'multi_asset',
                'description': 'Introduce multiple assets',
                'difficulty': 0.5,
                'max_leverage': 5,
                'assets': ['BTC/USD:USD', 'ETH/USD:USD'],
                'window_size': 100,
                'timesteps': 2_000_000
            },
            {
                'name': 'advanced_trading',
                'description': 'Higher leverage and more assets',
                'difficulty': 0.8,
                'max_leverage': 10,
                'assets': ['BTC/USD:USD', 'ETH/USD:USD', 'SOL/USD:USD'],
                'window_size': 200,
                'timesteps': 3_000_000
            },
            {
                'name': 'expert',
                'description': 'Full leverage and all features',
                'difficulty': 1.0,
                'max_leverage': max_leverage,
                'assets': ['BTC/USD:USD', 'ETH/USD:USD', 'SOL/USD:USD', 'AVAX/USD:USD'],
                'window_size': 500,
                'timesteps': 5_000_000
            }
        ]
        
        self.current_stage = 0
        
    def create_envs(self, data: pd.DataFrame) -> Tuple[VecNormalize, Dict]:
        """Create vectorized environments for current stage"""
        stage = self.stages[self.current_stage]
        
        def make_env(rank: int):
            def _init():
                env = InstitutionalPerpetualEnv(
                    df=data,
                    initial_balance=self.initial_balance,
                    max_leverage=stage['max_leverage'],
                    window_size=stage['window_size']
                )
                return env
            return _init
            
        env = SubprocVecEnv([make_env(i) for i in range(self.n_envs)])
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
        
        return env, stage
        
    def next_stage(self) -> bool:
        """Advance to next curriculum stage"""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            return True
        return False
        
    def get_stage_info(self) -> Dict:
        """Get current stage information"""
        return self.stages[self.current_stage]

class DataScheduler:
    """Manages data sampling and progression"""
    def __init__(
        self,
        data: pd.DataFrame,
        initial_window: int = 1000,
        max_window: int = None,
        window_growth_rate: float = 1.2  # Reduced from 1.5 to prevent too rapid growth
    ):
        self.data = data
        total_data_length = len(data)
        
        # Ensure initial window is not too large
        self.initial_window = min(initial_window, total_data_length // 2)
        
        # Set max window to be 80% of data length if not specified
        self.max_window = min(max_window or total_data_length, int(total_data_length * 0.8))
        self.window_growth_rate = window_growth_rate
        self.current_window = self.initial_window
        
        logger.info(f"DataScheduler initialized with: total_data_length={total_data_length}, "
                   f"initial_window={self.initial_window}, max_window={self.max_window}")
        
    def get_train_window(self, stage: int) -> pd.DataFrame:
        """Get training data window for current stage"""
        # Calculate window size with safety checks
        window_size = min(
            int(self.initial_window * (self.window_growth_rate ** stage)),
            self.max_window
        )
        
        # Ensure window size doesn't exceed data length
        window_size = min(window_size, len(self.data) - 1)
        self.current_window = window_size
        
        # Randomly select starting point
        max_start = max(0, len(self.data) - window_size - 1)  # Ensure max_start is at least 0
        start = 0 if max_start == 0 else np.random.randint(0, max_start)
        
        logger.info(f"Selected training window: stage={stage}, window_size={window_size}, "
                   f"start={start}, data_length={len(self.data)}")
        
        return self.data.iloc[start:start + window_size]
        
    def get_eval_window(self) -> pd.DataFrame:
        """Get evaluation data window"""
        # Use different time period for evaluation
        max_start = max(0, len(self.data) - self.current_window - 1)  # Ensure max_start is at least 0
        start = 0 if max_start == 0 else np.random.randint(0, max_start)
        
        logger.info(f"Selected evaluation window: window_size={self.current_window}, "
                   f"start={start}, data_length={len(self.data)}")
        
        return self.data.iloc[start:start + self.current_window]

class TrainingManager:
    """Manages the entire training process"""
    def __init__(
        self,
        data_manager,
        initial_balance: float = 1e6,
        max_leverage: float = 20,
        n_envs: int = 8,
        wandb_config: dict = None
    ):
        self.data_manager = data_manager
        self.curriculum = CurriculumScheduler(
            initial_balance=initial_balance,
            max_leverage=max_leverage,
            n_envs=n_envs
        )
        
        # Initialize wandb logger if config provided
        self.logger = None
        if wandb_config:
            self.logger = WandBLogger(
                config={
                    'initial_balance': initial_balance,
                    'max_leverage': max_leverage,
                    'n_envs': n_envs,
                    'curriculum_stages': self.curriculum.stages
                },
                project=wandb_config.get('project'),
                entity=wandb_config.get('entity')
            )
        
        # Load base features
        self.data = self.data_manager.load_feature_data('base_features')
        if self.data is None:
            raise ValueError("No feature data found. Please run data fetching and feature engineering first.")
        
        self.data_scheduler = DataScheduler(self.data)
        
    def train(self, model) -> None:
        """Execute full training curriculum"""
        try:
            stage = self.curriculum.get_stage_info()
            
            while True:
                # Get training data window
                train_data = self.data_scheduler.get_train_window(
                    self.curriculum.current_stage
                )
                
                # Create environments
                env, stage = self.curriculum.create_envs(train_data)
                
                # Log stage transition if logger exists
                if self.logger:
                    self.logger.log_metrics({
                        'curriculum/stage': stage['name'],
                        'curriculum/difficulty': stage['difficulty']
                    })
                
                # Train for stage's timesteps
                model.learn(
                    total_timesteps=stage['timesteps'],
                    callback=self._create_callbacks(stage)
                )
                
                # Validate performance
                val_metrics = self._validate(model)
                
                # Log validation metrics if logger exists
                if self.logger:
                    self.logger.log_metrics({
                        'stage': stage['name'],
                        'validation/sharpe': val_metrics['sharpe'],
                        'validation/returns': val_metrics['returns'],
                        'validation/drawdown': val_metrics['drawdown']
                    })
                
                # Check if we should advance to next stage
                if self._should_advance(val_metrics):
                    if not self.curriculum.next_stage():
                        break  # Training complete
                        
                # Save checkpoint
                model.save(f"checkpoints/stage_{stage['name']}")
                
                # Save training state
                self._save_training_state(stage, val_metrics)
                
        finally:
            if self.logger:
                self.logger.finish()
    
    def _save_training_state(self, stage: Dict, metrics: Dict):
        """Save training state for potential resumption"""
        state = {
            'stage': stage,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        pd.DataFrame([state]).to_parquet(f"checkpoints/training_state_{stage['name']}.parquet")
    
    def load_training_state(self, stage_name: str) -> Optional[Dict]:
        """Load training state for resumption"""
        try:
            state_file = f"checkpoints/training_state_{stage_name}.parquet"
            if os.path.exists(state_file):
                state = pd.read_parquet(state_file).iloc[0].to_dict()
                return state
            return None
        except Exception as e:
            logger.error(f"Error loading training state: {str(e)}")
            return None
        
    def _create_callbacks(self, stage: Dict) -> List:
        """Create stage-specific callbacks"""
        callbacks = []
        # Add custom callbacks here
        return callbacks
        
    def _validate(self, model) -> Dict:
        """Evaluate model on validation data"""
        val_data = self.data_scheduler.get_eval_window()
        env, _ = self.curriculum.create_envs(val_data)
        
        # Run validation episodes
        returns = []
        drawdowns = []
        
        obs = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            
            if done:
                returns.append(info[0]['returns'])
                drawdowns.append(info[0]['drawdown'])
                
        # Calculate metrics
        returns = np.array(returns)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-6)
        
        return {
            'sharpe': sharpe,
            'returns': np.mean(returns),
            'drawdown': np.mean(drawdowns)
        }
        
    def _should_advance(self, metrics: Dict) -> bool:
        """Determine if we should advance to next stage"""
        # Define stage completion criteria
        stage = self.curriculum.get_stage_info()
        
        if stage['name'] == 'basic_trading':
            return metrics['sharpe'] > 1.0 and metrics['returns'] > 0.1
            
        elif stage['name'] == 'multi_asset':
            return metrics['sharpe'] > 1.5 and metrics['returns'] > 0.15
            
        elif stage['name'] == 'advanced_trading':
            return metrics['sharpe'] > 2.0 and metrics['returns'] > 0.2
            
        return False  # Don't advance from final stage

# Example configuration
DEFAULT_CURRICULUM = {
    "beginner": {
        "market_volatility": 0.5,
        "allowed_leverage": 5,
        "liquidation_multiplier": 0.8,
        "promotion_threshold": 1.2,
        "demotion_threshold": 0.8
    },
    "intermediate": {
        "market_volatility": 1.0,
        "allowed_leverage": 10,
        "liquidation_multiplier": 1.0,
        "promotion_threshold": 1.5,
        "demotion_threshold": 1.0
    },
    "expert": {
        "market_volatility": 2.0,
        "allowed_leverage": 20,
        "liquidation_multiplier": 1.2,
        "promotion_threshold": 2.0,
        "demotion_threshold": 1.2
    }
}

def create_default_curriculum() -> List[CurriculumStage]:
    """Create default curriculum stages"""
    return [
        CurriculumStage(
            name="Basics",
            description="Learn basic trading with low risk",
            difficulty=0.2,
            env_params={
                'max_leverage': 2,
                'transaction_fee': 0.0002,
                'funding_fee_multiplier': 0.5,
                'max_drawdown': 0.1
            },
            required_metrics={
                'sharpe': 0.5,
                'max_drawdown': -0.05
            },
            duration=100000
        ),
        CurriculumStage(
            name="Intermediate",
            description="Increase complexity with moderate risk",
            difficulty=0.5,
            env_params={
                'max_leverage': 5,
                'transaction_fee': 0.0004,
                'funding_fee_multiplier': 0.8,
                'max_drawdown': 0.15
            },
            required_metrics={
                'sharpe': 1.0,
                'max_drawdown': -0.1
            },
            duration=200000
        ),
        CurriculumStage(
            name="Advanced",
            description="Full complexity with realistic conditions",
            difficulty=0.8,
            env_params={
                'max_leverage': 10,
                'transaction_fee': 0.0004,
                'funding_fee_multiplier': 1.0,
                'max_drawdown': 0.2
            },
            required_metrics={
                'sharpe': 1.5,
                'max_drawdown': -0.15
            },
            duration=300000
        ),
        CurriculumStage(
            name="Expert",
            description="Maximum complexity with full risk",
            difficulty=1.0,
            env_params={
                'max_leverage': 20,
                'transaction_fee': 0.0004,
                'funding_fee_multiplier': 1.0,
                'max_drawdown': 0.3
            },
            required_metrics={
                'sharpe': 2.0,
                'max_drawdown': -0.2
            },
            duration=400000
        )
    ]