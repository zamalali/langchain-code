import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import os
import logging
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("router.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()

class TaskComplexity(Enum):
    TRIVIAL = 1
    SIMPLE = 2
    MODERATE = 3
    COMPLEX = 4
    EXPERT = 5

class TaskType(Enum):
    CODE_COMPLETION = "completion"
    CODE_GENERATION = "generation"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    TESTING = "testing"
    MULTIMODAL = "multimodal"
    ANALYSIS = "analysis"

@dataclass
class LLMCapabilities:
    name: str
    provider: str
    api_key_env: str  
    max_context: int
    tokens_per_second: float
    cost_per_1k_input: float
    cost_per_1k_output: float
    strengths: List[TaskType]
    weaknesses: List[TaskType] = field(default_factory=list)
    multimodal: bool = False
    code_languages_strength: List[str] = field(default_factory=list)
    reasoning_capability: float = 1.0
    latency_ms: int = 1000
    reliability_score: float = 0.95

@dataclass
class TaskContext:
    task_type: TaskType
    estimated_complexity: TaskComplexity
    context_size: int
    file_count: int
    languages: List[str]
    time_sensitive: bool = False
    budget_constraint: Optional[float] = None
    quality_requirement: float = 0.8
    has_multimodal_input: bool = False
    user_preference: Optional[str] = None  # e.g., "gemini", "claude"

@dataclass
class RoutingDecision:
    selected_llm: str
    confidence: float
    reasoning: str
    context_vector: List[float]
    rule_based_score: float
    bandit_score: Optional[float] = None
    decision_source: str = "rule_based"

@dataclass
class PerformanceRecord:
    llm_name: str
    task_context: TaskContext
    context_vector: List[float]
    success: bool
    response_time: float
    cost: float
    quality_score: float
    user_satisfaction: float
    timestamp: float
    reward: float

class TaskAnalyzer:
    def __init__(self):
        self.complexity_indicators = {
            TaskComplexity.TRIVIAL: {
                'keywords': {'fix typo': 2.0, 'add comment': 1.5, 'format': 1.5, 'lint': 1.5, 'quick fix': 2.0},
                'max_files': 1,
                'max_lines': 10,
                'context_patterns': {'small': 1.0, 'quick': 1.0, 'simple': 1.0}
            },
            TaskComplexity.SIMPLE: {
                'keywords': {'add function': 2.0, 'simple': 1.5, 'basic': 1.5, 'quick': 1.0, 'single': 1.0},
                'max_files': 3,
                'max_lines': 100,
                'context_patterns': {'function': 1.0, 'method': 1.0, 'class': 1.0}
            },
            TaskComplexity.MODERATE: {
                'keywords': {'refactor': 2.0, 'reorganize': 1.5, 'multiple files': 2.0, 'update': 1.0, 'improve': 1.0},
                'max_files': 10,
                'max_lines': 1000,
                'context_patterns': {'module': 1.0, 'component': 1.0, 'service': 1.0}
            },
            TaskComplexity.COMPLEX: {
                'keywords': {'architecture': 2.0, 'design pattern': 2.0, 'system': 1.5, 'entire': 1.5, 'comprehensive': 1.5},
                'max_files': 50,
                'max_lines': 5000,
                'context_patterns': {'system': 1.0, 'application': 1.0, 'platform': 1.0}
            },
            TaskComplexity.EXPERT: {
                'keywords': {'optimize performance': 2.0, 'advanced': 1.5, 'algorithm': 2.0, 'machine learning': 2.0, 'complex': 1.5},
                'max_files': float('inf'),
                'max_lines': float('inf'),
                'context_patterns': {'optimization': 1.0, 'algorithm': 1.0, 'research': 1.0}
            }
        }
        
        self.task_patterns = {
            TaskType.CODE_COMPLETION: {'complete': 2.0, 'finish': 2.0, 'autocomplete': 1.5, 'suggestion': 1.5, 'continue': 1.5},
            TaskType.CODE_GENERATION: {'create': 2.0, 'generate': 2.0, 'build': 1.5, 'implement': 1.5, 'write': 1.5, 'develop': 1.5, 'make': 1.0},
            TaskType.REFACTORING: {'refactor': 2.0, 'restructure': 1.5, 'reorganize': 1.5, 'clean up': 1.5, 'improve': 1.0, 'modernize': 1.0},
            TaskType.DOCUMENTATION: {'document': 2.0, 'docs': 2.0, 'readme': 1.5, 'comments': 1.5, 'explain': 1.0, 'api doc': 1.5, 'guide': 1.0},
            TaskType.DEBUGGING: {'debug': 2.0, 'fix bug': 2.0, 'error': 1.5, 'issue': 1.5, 'problem': 1.5, 'broken': 1.5, 'typo': 1.0, 'fix': 1.0},
            TaskType.ARCHITECTURE: {'architecture': 2.0, 'design': 1.5, 'structure': 1.5, 'pattern': 1.5, 'schema': 1.0, 'blueprint': 1.0},
            TaskType.TESTING: {'test': 2.0, 'unit test': 2.0, 'integration': 1.5, 'coverage': 1.5, 'spec': 1.0, 'verify': 1.0},
            TaskType.MULTIMODAL: {'image': 2.0, 'screenshot': 2.0, 'ui': 1.5, 'design': 1.5, 'visual': 1.5, 'figma': 1.5, 'mockup': 1.5},
            TaskType.ANALYSIS: {'analyze': 2.0, 'review': 1.5, 'audit': 1.5, 'examine': 1.0, 'understand': 1.0, 'investigate': 1.0}
        }

    def analyze_task(self, user_input: str, context_info: Dict) -> TaskContext:
        user_input_lower = user_input.lower()
        
        task_type = self._detect_task_type(user_input_lower)
        complexity = self._estimate_complexity(user_input_lower, context_info)
        
        context_size = context_info.get('context_size', 0)
        file_count = context_info.get('file_count', 0)
        languages = context_info.get('languages', [])
        
        has_multimodal = (
            context_info.get('has_image', False) or
            any(keyword in user_input_lower for keyword in self.task_patterns[TaskType.MULTIMODAL])
        )
        
        time_sensitive = any(word in user_input_lower 
                           for word in ['quick', 'fast', 'urgent', 'asap', 'immediately', 'now'])
        
        # Detect user preference for specific provider
        user_preference = None
        if 'use gemini' in user_input_lower or 'google' in user_input_lower:
            user_preference = 'google'
        elif 'use claude' in user_input_lower or 'anthropic' in user_input_lower:
            user_preference = 'anthropic'

        return TaskContext(
            task_type=task_type,
            estimated_complexity=complexity,
            context_size=context_size,
            file_count=file_count,
            languages=languages,
            time_sensitive=time_sensitive,
            has_multimodal_input=has_multimodal,
            user_preference=user_preference
        )
    
    def _detect_task_type(self, user_input: str) -> TaskType:
        # Priority rules for task type detection
        if not user_input:  # Handle empty input
            return TaskType.ANALYSIS
        
        if any(word in user_input for word in self.task_patterns[TaskType.DEBUGGING]):
            return TaskType.DEBUGGING
        if any(word in user_input for word in self.task_patterns[TaskType.CODE_COMPLETION]):
            return TaskType.CODE_COMPLETION
        if any(word in user_input for word in self.task_patterns[TaskType.MULTIMODAL]):
            return TaskType.MULTIMODAL
        if any(word in user_input for word in ['entire', 'whole', 'all', 'comprehensive', 'codebase']):
            if any(word in user_input for word in self.task_patterns[TaskType.DOCUMENTATION]):
                return TaskType.DOCUMENTATION
            elif any(word in user_input for word in self.task_patterns[TaskType.ANALYSIS]):
                return TaskType.ANALYSIS
            elif any(word in user_input for word in self.task_patterns[TaskType.ARCHITECTURE]):
                return TaskType.ARCHITECTURE
        
        scores = {}
        for task_type, patterns in self.task_patterns.items():
            score = sum(weight for pattern, weight in patterns.items() if pattern in user_input)
            if score > 0:
                scores[task_type] = score
        
        return max(scores.items(), key=lambda x: x[1])[0] if scores else TaskType.ANALYSIS  
    
    def _estimate_complexity(self, user_input: str, context_info: Dict) -> TaskComplexity:
        file_count = context_info.get('file_count', 0)
        context_size = context_info.get('context_size', 0)
        
        complexity_scores = {}
        for complexity, indicators in self.complexity_indicators.items():
            score = 0
            for keyword, weight in indicators['keywords'].items():
                if keyword in user_input:
                    score += weight
            for pattern, weight in indicators.get('context_patterns', {}).items():
                if pattern in user_input:
                    score += weight
            complexity_scores[complexity] = score
        
        keyword_complexity = max(complexity_scores.items(), key=lambda x: x[1])[0] if any(complexity_scores.values()) else None
        
        size_complexity = TaskComplexity.TRIVIAL
        for complexity, indicators in self.complexity_indicators.items():
            if (file_count <= indicators['max_files'] and 
                context_size <= indicators['max_lines']):
                size_complexity = complexity
                break
        
        return max(keyword_complexity, size_complexity, key=lambda x: x.value) if keyword_complexity else size_complexity

class ContextualBandit:
    def __init__(self, llm_names: List[str], context_dim: int = 14, alpha: float = 0.5, decay_rate: float = 0.01):
        self.llm_names = llm_names
        self.context_dim = context_dim
        self.alpha = alpha
        self.decay_rate = decay_rate  # For decaying old performance data
        
        self.A = {llm: np.eye(context_dim) for llm in llm_names}
        self.b = {llm: np.zeros(context_dim) for llm in llm_names}
        self.theta = {llm: np.zeros(context_dim) for llm in llm_names}
        
        self.total_interactions = 0
        self.arm_counts = {llm: 0 for llm in llm_names}
        self.performance_history = []
    
    def _compute_context_vector(self, task_context: TaskContext) -> np.ndarray:
        vector = np.zeros(self.context_dim)
        
        task_types = list(TaskType)
        for idx, task_type in enumerate(task_types[:9]):  # Reserve first 9 dims for task types
            if task_context.task_type == task_type:
                vector[idx] = 1.0
        
        vector[9] = min(task_context.estimated_complexity.value / 5.0, 1.0)  # Complexity
        vector[10] = min(task_context.context_size / 100000.0, 1.0)  # Context size
        vector[11] = 1.0 if task_context.time_sensitive else 0.0  # Time sensitivity
        vector[12] = 1.0 if task_context.has_multimodal_input else 0.0  # Multimodal
        vector[13] = min(task_context.file_count / 100.0, 1.0)  # File count
        
        return vector
    
    def select_arm(self, task_context: TaskContext) -> Tuple[str, float]:
        context_vector = self._compute_context_vector(task_context)
        
        # Cold start: round-robin until each LLM is tried at least twice
        if self.total_interactions < len(self.llm_names) * 2:
            selected_llm = self.llm_names[self.total_interactions % len(self.llm_names)]
            logger.info(f"Cold start: Selecting {selected_llm} (interaction {self.total_interactions + 1})")
            return selected_llm, 0.5
        
        ucb_values = {}
        for llm in self.llm_names:
            try:
                self.theta[llm] = np.linalg.solve(self.A[llm], self.b[llm])
            except np.linalg.LinAlgError:
                self.theta[llm] = np.zeros(self.context_dim)
            
            try:
                confidence_interval = self.alpha * np.sqrt(
                    context_vector.T @ np.linalg.inv(self.A[llm]) @ context_vector
                )
            except np.linalg.LinAlgError:
                confidence_interval = self.alpha
            
            expected_reward = context_vector.T @ self.theta[llm]
            ucb_values[llm] = expected_reward + confidence_interval
        
        selected_llm = max(ucb_values.items(), key=lambda x: x[1])[0]
        confidence = min(max(ucb_values[selected_llm], 0.0), 1.0)
        logger.debug(f"Bandit selected {selected_llm} with confidence {confidence:.3f}")
        return selected_llm, confidence
    
    def update_arm(self, llm_name: str, context_vector: np.ndarray, reward: float):
        # Apply decay to existing matrices to prioritize recent data
        self.A[llm_name] = self.A[llm_name] * (1 - self.decay_rate) + np.outer(context_vector, context_vector)
        self.b[llm_name] = self.b[llm_name] * (1 - self.decay_rate) + reward * context_vector
        self.arm_counts[llm_name] += 1
        self.total_interactions += 1
    
    def has_sufficient_data(self, task_context: TaskContext, min_interactions: int = 20) -> bool:
        return self.total_interactions >= min_interactions
    
    def save_state(self, filepath: str):
        state = {
            'A': {llm: A.tolist() for llm, A in self.A.items()},
            'b': {llm: b.tolist() for llm, b in self.b.items()},
            'theta': {llm: theta.tolist() for llm, theta in self.theta.items()},
            'total_interactions': self.total_interactions,
            'arm_counts': self.arm_counts,
            'performance_history': [
                {
                    'llm_name': r.llm_name,
                    'task_context': r.task_context.__dict__,
                    'context_vector': r.context_vector,
                    'success': r.success,
                    'response_time': r.response_time,
                    'cost': r.cost,
                    'quality_score': r.quality_score,
                    'user_satisfaction': r.user_satisfaction,
                    'timestamp': r.timestamp,
                    'reward': r.reward
                } for r in self.performance_history
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.A = {llm: np.array(A) for llm, A in state['A'].items()}
            self.b = {llm: np.array(b) for llm, b in state['b'].items()}
            self.theta = {llm: np.array(theta) for llm, theta in state['theta'].items()}
            self.total_interactions = state['total_interactions']
            self.arm_counts = state['arm_counts']
            self.performance_history = [
                PerformanceRecord(
                    llm_name=r['llm_name'],
                    task_context=TaskContext(**r['task_context']),
                    context_vector=r['context_vector'],
                    success=r['success'],
                    response_time=r['response_time'],
                    cost=r['cost'],
                    quality_score=r['quality_score'],
                    user_satisfaction=r['user_satisfaction'],
                    timestamp=r['timestamp'],
                    reward=r['reward']
                ) for r in state.get('performance_history', [])
            ]

class RuleBasedRouter:
    def __init__(self, llm_registry: Dict[str, LLMCapabilities]):
        self.llm_registry = llm_registry
    
    def calculate_llm_score(self, capabilities: LLMCapabilities, task_context: TaskContext, 
                        user_preferences: Optional[Dict] = None) -> float:
        score = 0.0
        
        if task_context.task_type in capabilities.strengths:
            score += 0.35
        elif task_context.task_type in capabilities.weaknesses:
            score -= 0.20
        
        if task_context.context_size > capabilities.max_context:
            score -= 0.40
        elif task_context.context_size > 50000:
            if capabilities.max_context > 500000:
                score += 0.15  
            elif capabilities.max_context > 100000:
                score += 0.10
            else:
                score -= 0.20
        else:
            context_utilization = task_context.context_size / capabilities.max_context
            score += 0.15 * (1.0 - context_utilization)
        
        # Time sensitivity
        if task_context.time_sensitive:
            if capabilities.tokens_per_second > 300:
                score += 0.20
            elif capabilities.tokens_per_second > 100:
                score += 0.10
            else:
                score -= 0.15
        
        # Complexity requirements (increased weight for high reasoning)
        if task_context.estimated_complexity.value >= 4:
            if capabilities.reasoning_capability > 0.9:
                score += 0.25  # Increased from 0.15 for complex/expert tasks
            elif capabilities.reasoning_capability > 0.8:
                score += 0.15  # Increased from 0.10
            else:
                score -= 0.10  # Penalize low reasoning for complex tasks
        else:
            if capabilities.reasoning_capability < 0.8:
                score += 0.05
        
        # Model-specific bonuses
        if task_context.context_size > 80000 and capabilities.name == 'gemini-1.5-pro':
            score += 0.15  # Reduced from 0.30
        if (task_context.time_sensitive and 
            task_context.estimated_complexity.value <= 2 and
            capabilities.name == 'llama-3.1-8b-instant'):
            score += 0.40
        if task_context.task_type == TaskType.DEBUGGING and 'deepseek' in capabilities.name:
            score += 0.25
        if task_context.estimated_complexity.value <= 2 and capabilities.name == 'claude-3.5-sonnet':
            score -= 0.25
        
        # Multimodal support
        if task_context.has_multimodal_input:
            if capabilities.multimodal:
                score += 0.15
            else:
                score -= 0.50
        
        # Language compatibility
        if any(lang in capabilities.code_languages_strength for lang in task_context.languages):
            score += 0.05
        
        # User preferences
        if user_preferences:
            if user_preferences.get('preferred_llm') == capabilities.name:
                score += 0.20
            elif user_preferences.get('avoid_llm') == capabilities.name:
                score -= 0.30
        if task_context.user_preference and capabilities.provider == task_context.user_preference:
            score += 0.25
        
        return max(0.0, min(1.0, score))
    
    def route_request(self, task_context: TaskContext, user_preferences: Optional[Dict] = None) -> Tuple[str, float]:
        scores = {}
        for llm_name, capabilities in self.llm_registry.items():
            score = self.calculate_llm_score(capabilities, task_context, user_preferences)
            scores[llm_name] = score
        
        if not scores:
            logger.error("No available LLMs for routing")
            raise ValueError("No available LLMs for routing")
        
        best_llm = max(scores.items(), key=lambda x: x[1])
        logger.debug(f"Rule-based routing selected {best_llm[0]} with score {best_llm[1]:.3f}")
        return best_llm[0], best_llm[1]

class HybridIntelligentRouter:
    def __init__(self, registry_file: str = "llm_registry.yaml", state_file: str = "router_state.json"):
        self.registry_file = registry_file
        self.state_file = state_file
        self.llm_registry = self._load_llm_registry()
        self.available_llms = self._filter_available_llms()
        self.task_analyzer = TaskAnalyzer()
        
        self.rule_based_router = RuleBasedRouter(self.available_llms)
        self.bandit_router = ContextualBandit(
            list(self.available_llms.keys()),
            context_dim=14,
            alpha=0.5,
            decay_rate=0.01
        )
        
        self.bandit_router.load_state(self.state_file)
        
        self.bandit_confidence_threshold = 0.6
        self.min_bandit_interactions = 10
        self.performance_history = []
        self.default_llm = 'gemini-1.5-pro'
    
    def _load_llm_registry(self) -> Dict[str, LLMCapabilities]:
        """Load LLM registry from a YAML file or use default if file not found"""
        default_registry = {
            'claude-3.5-sonnet': {
                'name': 'claude-3.5-sonnet',
                'provider': 'anthropic',
                'api_key_env': 'ANTHROPIC_API_KEY',
                'max_context': 200000,
                'tokens_per_second': 50.0,
                'cost_per_1k_input': 3.0,
                'cost_per_1k_output': 15.0,
                'strengths': ['CODE_GENERATION', 'REFACTORING', 'ARCHITECTURE', 'ANALYSIS'],
                'code_languages_strength': ['python', 'javascript', 'typescript', 'rust', 'go'],
                'reasoning_capability': 0.95,
                'latency_ms': 1200,
                'reliability_score': 0.95,
                'multimodal': True
            },
            'gpt-4-turbo': {
                'name': 'gpt-4-turbo',
                'provider': 'openai',
                'api_key_env': 'OPENAI_API_KEY',
                'max_context': 128000,
                'tokens_per_second': 40.0,
                'cost_per_1k_input': 10.0,
                'cost_per_1k_output': 30.0,
                'strengths': ['CODE_GENERATION', 'DOCUMENTATION', 'ARCHITECTURE'],
                'code_languages_strength': ['python', 'javascript', 'java', 'c++'],
                'reasoning_capability': 0.90,
                'latency_ms': 1500,
                'reliability_score': 0.92,
                'multimodal': True
            },
            'llama-3.1-8b-instant': {
                'name': 'llama-3.1-8b-instant',
                'provider': 'groq',
                'api_key_env': 'GROQ_API_KEY',
                'max_context': 8192,
                'tokens_per_second': 500.0,
                'cost_per_1k_input': 0.59,
                'cost_per_1k_output': 0.79,
                'strengths': ['CODE_COMPLETION', 'CODE_GENERATION', 'DEBUGGING'],
                'code_languages_strength': ['python', 'javascript'],
                'reasoning_capability': 0.75,
                'latency_ms': 200,
                'reliability_score': 0.88
            },
            'deepseek-coder': {
                'name': 'deepseek-coder',
                'provider': 'deepseek',
                'api_key_env': 'DEEPSEEK_API_KEY',
                'max_context': 16384,
                'tokens_per_second': 100.0,
                'cost_per_1k_input': 0.14,
                'cost_per_1k_output': 0.28,
                'strengths': ['CODE_GENERATION', 'DEBUGGING', 'REFACTORING'],
                'code_languages_strength': ['python', 'java', 'cpp', 'javascript', 'go'],
                'reasoning_capability': 0.80,
                'latency_ms': 800,
                'reliability_score': 0.90
            },
            'gemini-1.5-pro': {
                'name': 'gemini-1.5-pro',
                'provider': 'google',
                'api_key_env': 'GOOGLE_API_KEY',
                'max_context': 1000000,
                'tokens_per_second': 35.0,
                'cost_per_1k_input': 1.25,
                'cost_per_1k_output': 5.0,
                'strengths': ['ANALYSIS', 'DOCUMENTATION', 'MULTIMODAL', 'CODE_GENERATION'],
                'code_languages_strength': ['python', 'javascript', 'java'],
                'reasoning_capability': 0.85,
                'latency_ms': 1800,
                'reliability_score': 0.89,
                'multimodal': True
            }
        }
        
        if Path(self.registry_file).exists():
            try:
                with open(self.registry_file, 'r') as f:
                    registry_data = yaml.safe_load(f)
                if not registry_data:
                    logger.warning(f"Registry file {self.registry_file} is empty, using default registry")
                    registry_data = default_registry
            except Exception as e:
                logger.error(f"Failed to load registry from {self.registry_file}: {e}, using default registry")
                registry_data = default_registry
        else:
            logger.info(f"No registry file found at {self.registry_file}, using default registry")
            registry_data = default_registry
        
        registry = {}
        for name, config in registry_data.items():
            registry[name] = LLMCapabilities(
                name=config['name'],
                provider=config['provider'],
                api_key_env=config['api_key_env'],  # Add this field
                max_context=config['max_context'],
                tokens_per_second=config['tokens_per_second'],
                cost_per_1k_input=config['cost_per_1k_input'],
                cost_per_1k_output=config['cost_per_1k_output'],
                strengths=[TaskType[s] for s in config['strengths']],
                weaknesses=[TaskType[s] for s in config.get('weaknesses', [])],
                multimodal=config.get('multimodal', False),
                code_languages_strength=config.get('code_languages_strength', []),
                reasoning_capability=config.get('reasoning_capability', 1.0),
                latency_ms=config.get('latency_ms', 1000),
                reliability_score=config.get('reliability_score', 0.95)
            )
        return registry
    
    def _filter_available_llms(self) -> Dict[str, LLMCapabilities]:
        """Filter LLMs based on available API keys"""
        available = {}
        for name, capabilities in self.llm_registry.items():
            api_key = os.getenv(capabilities.api_key_env)
            if api_key:
                available[name] = capabilities
                logger.debug(f"LLM {name} included: API key found for {capabilities.provider}")
            else:
                logger.debug(f"LLM {name} excluded: No API key found for {capabilities.api_key_env}")
        
        if not available:
            logger.warning(f"No valid API keys found, defaulting to {self.default_llm}")
            available[self.default_llm] = self.llm_registry.get(self.default_llm, self.llm_registry['gemini-1.5-pro'])
        
        return available
    
    def update_available_llms(self):
        """Refresh the available LLMs and update bandit router"""
        self.available_llms = self._filter_available_llms()
        self.bandit_router = ContextualBandit(
            list(self.available_llms.keys()),
            context_dim=14,
            alpha=0.5,
            decay_rate=0.01
        )
        self.bandit_router.load_state(self.state_file)
        self.rule_based_router = RuleBasedRouter(self.available_llms)
        logger.info(f"Updated available LLMs: {list(self.available_llms.keys())}")
    
    def route_request(self, user_input: str, context_info: Dict, 
                     user_preferences: Optional[Dict] = None) -> RoutingDecision:
        """Route request using hybrid approach with API key filtering"""
        task_context = self.task_analyzer.analyze_task(user_input, context_info)
        context_vector = self.bandit_router._compute_context_vector(task_context)
        
        # Apply user preference for provider if specified
        if task_context.user_preference:
            filtered_llms = {
                name: caps for name, caps in self.available_llms.items()
                if caps.provider == task_context.user_preference
            }
            if not filtered_llms:
                logger.warning(f"No available LLMs for preferred provider {task_context.user_preference}")
                filtered_llms = self.available_llms
        else:
            filtered_llms = self.available_llms
        
        if not filtered_llms:
            logger.error("No LLMs available after filtering")
            raise ValueError(f"No LLMs available. Please provide at least one valid API key.")
        
        self.rule_based_router.llm_registry = filtered_llms
        rule_llm, rule_confidence = self.rule_based_router.route_request(task_context, user_preferences)
        
        bandit_llm, bandit_confidence = None, None
        if self.bandit_router.has_sufficient_data(task_context, self.min_bandit_interactions):
            self.bandit_router.llm_names = list(filtered_llms.keys())
            bandit_llm, bandit_confidence = self.bandit_router.select_arm(task_context)
        
        if (bandit_llm and 
            bandit_confidence > self.bandit_confidence_threshold and
            bandit_confidence > rule_confidence):
            selected_llm = bandit_llm
            confidence = bandit_confidence
            decision_source = "bandit"
            reasoning = f"Bandit selection based on learned performance patterns (confidence: {bandit_confidence:.3f})"
        else:
            selected_llm = rule_llm
            confidence = rule_confidence
            decision_source = "rule_based"
            reasoning = self._generate_rule_based_reasoning(selected_llm, task_context)
        
        logger.info(f"Routed to {selected_llm} (source: {decision_source}, confidence: {confidence:.3f}, reasoning: {reasoning})")
        
        return RoutingDecision(
            selected_llm=selected_llm,
            confidence=confidence,
            reasoning=reasoning,
            context_vector=context_vector.tolist(),
            rule_based_score=rule_confidence,
            bandit_score=bandit_confidence,
            decision_source=decision_source
        )
    
    def _generate_rule_based_reasoning(self, selected_llm: str, task_context: TaskContext) -> str:
        capabilities = self.available_llms[selected_llm]
        reasons = []
        
        if task_context.task_type in capabilities.strengths:
            reasons.append(f"Optimized for {task_context.task_type.value} tasks")
        if capabilities.reasoning_capability > 0.9:
            reasons.append("High reasoning capability for complex tasks")
        if task_context.time_sensitive and capabilities.tokens_per_second > 200:
            reasons.append("Fast response time for time-sensitive task")
        if task_context.context_size > 50000 and capabilities.max_context > 100000:
            reasons.append("Large context window for comprehensive analysis")
        if task_context.has_multimodal_input and capabilities.multimodal:
            reasons.append("Multimodal support for image/visual inputs")
        if task_context.task_type == TaskType.DEBUGGING and 'deepseek' in selected_llm:
            reasons.append("Specialized for debugging and code analysis")
        if task_context.user_preference and capabilities.provider == task_context.user_preference:
            reasons.append(f"Matches user preference for {task_context.user_preference}")
        if not reasons:
            reasons.append("Best overall match for task requirements")
        
        return f"Selected {selected_llm}: " + ", ".join(reasons)
    
    def record_performance(self, llm_name: str, task_context: TaskContext, 
                          context_vector: List[float], success: bool, 
                          response_time: float, cost: float, 
                          quality_score: float, user_satisfaction: float):
        """Record performance and update bandit"""
        reward = (
            0.4 * quality_score +
            0.3 * user_satisfaction +
            0.2 * (1.0 if success else 0.0) +
            0.1 * max(0, 1.0 - response_time / 10.0)
        )
        
        cost_penalty = min(cost / 0.10, 1.0)
        reward = reward * (1.0 - 0.1 * cost_penalty)
        
        self.bandit_router.update_arm(llm_name, np.array(context_vector), reward)
        
        perf_record = PerformanceRecord(
            llm_name=llm_name,
            task_context=task_context,
            context_vector=context_vector,
            success=success,
            response_time=response_time,
            cost=cost,
            quality_score=quality_score,
            user_satisfaction=user_satisfaction,
            timestamp=time.time(),
            reward=reward
        )
        self.performance_history.append(perf_record)
        
        if len(self.performance_history) % 10 == 0:
            self.bandit_router.save_state(self.state_file)
            logger.info(f"Saved router state to {self.state_file}")