import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import os

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
    """Define capabilities and characteristics of each LLM"""
    name: str
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
    """Context information about the current task"""
    task_type: TaskType
    estimated_complexity: TaskComplexity
    context_size: int
    file_count: int
    languages: List[str]
    time_sensitive: bool = False
    budget_constraint: Optional[float] = None
    quality_requirement: float = 0.8  
    has_multimodal_input: bool = False
    user_preference: Optional[str] = None

@dataclass
class RoutingDecision:
    """Result of routing decision with metadata"""
    selected_llm: str
    confidence: float
    reasoning: str
    context_vector: List[float]
    rule_based_score: float
    bandit_score: Optional[float] = None
    decision_source: str = "rule_based" 

@dataclass
class PerformanceRecord:
    """Record of LLM performance for a specific task"""
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
    """Analyzes user input to determine task characteristics"""
    
    def __init__(self):
        self.complexity_indicators = {
            TaskComplexity.TRIVIAL: {
                'keywords': ['fix typo', 'add comment', 'format', 'lint', 'quick fix'],
                'max_files': 1,
                'max_lines': 10,
                'context_patterns': ['small', 'quick', 'simple']
            },
            TaskComplexity.SIMPLE: {
                'keywords': ['add function', 'simple', 'basic', 'quick', 'single'],
                'max_files': 3,
                'max_lines': 100,
                'context_patterns': ['function', 'method', 'class']
            },
            TaskComplexity.MODERATE: {
                'keywords': ['refactor', 'reorganize', 'multiple files', 'update', 'improve'],
                'max_files': 10,
                'max_lines': 1000,
                'context_patterns': ['module', 'component', 'service']
            },
            TaskComplexity.COMPLEX: {
                'keywords': ['architecture', 'design pattern', 'system', 'entire', 'comprehensive'],
                'max_files': 50,
                'max_lines': 5000,
                'context_patterns': ['system', 'application', 'platform']
            },
            TaskComplexity.EXPERT: {
                'keywords': ['optimize performance', 'advanced', 'algorithm', 'machine learning', 'complex'],
                'max_files': float('inf'),
                'max_lines': float('inf'),
                'context_patterns': ['optimization', 'algorithm', 'research']
            }
        }
        
        self.task_patterns = {
            TaskType.CODE_COMPLETION: [
                'complete', 'finish', 'autocomplete', 'suggestion', 'continue'
            ],
            TaskType.CODE_GENERATION: [
                'create', 'generate', 'build', 'implement', 'write', 'develop', 'make'
            ],
            TaskType.REFACTORING: [
                'refactor', 'restructure', 'reorganize', 'clean up', 'improve', 'modernize'
            ],
            TaskType.DOCUMENTATION: [
                'document', 'docs', 'readme', 'comments', 'explain', 'api doc', 'guide'
            ],
            TaskType.DEBUGGING: [
                'debug', 'fix bug', 'error', 'issue', 'problem', 'broken', 'typo', 'fix'
            ],
            TaskType.ARCHITECTURE: [
                'architecture', 'design', 'structure', 'pattern', 'schema', 'blueprint'
            ],
            TaskType.TESTING: [
                'test', 'unit test', 'integration', 'coverage', 'spec', 'verify'
            ],
            TaskType.MULTIMODAL: [
                'image', 'screenshot', 'ui', 'design', 'visual', 'figma', 'mockup'
            ],
            TaskType.ANALYSIS: [
                'analyze', 'review', 'audit', 'examine', 'understand', 'investigate'
            ]
        }
    
    def analyze_task(self, user_input: str, context_info: Dict) -> TaskContext:
        """Analyze user input and context to determine task characteristics"""
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
        
        return TaskContext(
            task_type=task_type,
            estimated_complexity=complexity,
            context_size=context_size,
            file_count=file_count,
            languages=languages,
            time_sensitive=time_sensitive,
            has_multimodal_input=has_multimodal
        )
    
    def _detect_task_type(self, user_input: str) -> TaskType:
        """Enhanced task type detection with priority rules"""
        
        if any(word in user_input for word in ['fix', 'bug', 'error', 'issue', 'debug', 'typo', 'broken']):
            return TaskType.DEBUGGING
        
        if any(word in user_input for word in ['complete', 'finish', 'continue']):
            return TaskType.CODE_COMPLETION
            
        if any(word in user_input for word in ['image', 'screenshot', 'ui', 'design', 'visual', 'figma', 'mockup']):
            return TaskType.MULTIMODAL
        
        if any(word in user_input for word in ['entire', 'whole', 'all', 'comprehensive', 'codebase']):
            if any(word in user_input for word in ['document', 'docs', 'readme']):
                return TaskType.DOCUMENTATION
            elif any(word in user_input for word in ['analyze', 'review', 'audit']):
                return TaskType.ANALYSIS
            elif any(word in user_input for word in ['architecture', 'design', 'structure']):
                return TaskType.ARCHITECTURE
        
        scores = {}
        for task_type, patterns in self.task_patterns.items():
            score = sum(1 for pattern in patterns if pattern in user_input)
            if score > 0:
                scores[task_type] = score
        
        if not scores:
            return TaskType.CODE_GENERATION  
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _estimate_complexity(self, user_input: str, context_info: Dict) -> TaskComplexity:
        """Enhanced complexity estimation"""
        file_count = context_info.get('file_count', 0)
        context_size = context_info.get('context_size', 0)
        
        complexity_scores = {}
        for complexity, indicators in self.complexity_indicators.items():
            score = 0
            for keyword in indicators['keywords']:
                if keyword in user_input:
                    score += 2
            for pattern in indicators.get('context_patterns', []):
                if pattern in user_input:
                    score += 1
            complexity_scores[complexity] = score
        
        keyword_complexity = max(complexity_scores.items(), key=lambda x: x[1])[0] if any(complexity_scores.values()) else None
        
        size_complexity = TaskComplexity.TRIVIAL
        for complexity, indicators in self.complexity_indicators.items():
            if (file_count <= indicators['max_files'] and 
                context_size <= indicators['max_lines']):
                size_complexity = complexity
                break
        
        if keyword_complexity:
            return max(keyword_complexity, size_complexity, key=lambda x: x.value)
        
        return size_complexity

class ContextualBandit:
    """Contextual Multi-Armed Bandit for adaptive LLM selection"""
    
    def __init__(self, llm_names: List[str], context_dim: int = 10, alpha: float = 1.0):
        self.llm_names = llm_names
        self.context_dim = context_dim
        self.alpha = alpha 
        
        self.A = {llm: np.eye(context_dim) for llm in llm_names}  
        self.b = {llm: np.zeros(context_dim) for llm in llm_names} 
        self.theta = {llm: np.zeros(context_dim) for llm in llm_names} 
        
        self.total_interactions = 0
        self.arm_counts = {llm: 0 for llm in llm_names}
        self.performance_history = []
        
    def _compute_context_vector(self, task_context: TaskContext) -> np.ndarray:
        """Convert task context to numerical vector"""
        vector = np.zeros(self.context_dim)
        
        task_types = list(TaskType)
        if len(task_types) <= 9:
            try:
                task_idx = task_types.index(task_context.task_type)
                vector[task_idx] = 1.0
            except ValueError:
                pass
        
        if self.context_dim > 9:
            vector[9] = min(task_context.estimated_complexity.value / 5.0, 1.0)  # Complexity (normalized)
        if self.context_dim > 10:
            vector[10] = min(task_context.context_size / 100000.0, 1.0)  # Context size (normalized)
        if self.context_dim > 11:
            vector[11] = 1.0 if task_context.time_sensitive else 0.0  # Time sensitivity
        if self.context_dim > 12:
            vector[12] = 1.0 if task_context.has_multimodal_input else 0.0  # Multimodal
        if self.context_dim > 13:
            vector[13] = min(task_context.file_count / 100.0, 1.0)  # File count (normalized)
        
        return vector
    
    def select_arm(self, task_context: TaskContext) -> Tuple[str, float]:
        """Select LLM using LinUCB algorithm"""
        context_vector = self._compute_context_vector(task_context)
        
        if self.total_interactions < len(self.llm_names) * 2:  # Cold start
            selected_llm = self.llm_names[self.total_interactions % len(self.llm_names)]
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
            
            # UCB value
            expected_reward = context_vector.T @ self.theta[llm]
            ucb_values[llm] = expected_reward + confidence_interval
        
        selected_llm = max(ucb_values.items(), key=lambda x: x[1])[0]
        confidence = min(max(ucb_values[selected_llm], 0.0), 1.0)
        
        return selected_llm, confidence
    
    def update_arm(self, llm_name: str, context_vector: np.ndarray, reward: float):
        """Update bandit parameters after observing reward"""
        self.A[llm_name] += np.outer(context_vector, context_vector)
        self.b[llm_name] += reward * context_vector
        self.arm_counts[llm_name] += 1
        self.total_interactions += 1
    
    def has_sufficient_data(self, task_context: TaskContext, min_interactions: int = 20) -> bool:
        """Check if bandit has enough data to make confident decisions"""
        return self.total_interactions >= min_interactions
    
    def save_state(self, filepath: str):
        """Save bandit state to file"""
        state = {
            'A': {llm: A.tolist() for llm, A in self.A.items()},
            'b': {llm: b.tolist() for llm, b in self.b.items()},
            'theta': {llm: theta.tolist() for llm, theta in self.theta.items()},
            'total_interactions': self.total_interactions,
            'arm_counts': self.arm_counts,
            'performance_history': self.performance_history
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load bandit state from file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.A = {llm: np.array(A) for llm, A in state['A'].items()}
            self.b = {llm: np.array(b) for llm, b in state['b'].items()}
            self.theta = {llm: np.array(theta) for llm, theta in state['theta'].items()}
            self.total_interactions = state['total_interactions']
            self.arm_counts = state['arm_counts']
            self.performance_history = state.get('performance_history', [])

class RuleBasedRouter:
    """Enhanced rule-based router for intelligent LLM selection"""
    
    def __init__(self, llm_registry: Dict[str, LLMCapabilities]):
        self.llm_registry = llm_registry
        
    def calculate_llm_score(self, capabilities: LLMCapabilities, 
                           task_context: TaskContext, 
                           user_preferences: Optional[Dict] = None) -> float:
        """Calculate composite score for LLM suitability"""
        score = 0.0
        
        if task_context.task_type in capabilities.strengths:
            score += 0.35
        elif task_context.task_type in capabilities.weaknesses:
            score -= 0.20
        
        if task_context.context_size > 50000:  
            if capabilities.max_context > 500000:  
                score += 0.25
            elif capabilities.max_context > 100000:   
                score += 0.15
            else:
                score -= 0.30 
        elif task_context.context_size <= capabilities.max_context:
            context_utilization = task_context.context_size / capabilities.max_context
            score += 0.15 * (1.0 - context_utilization)
        else:
            score -= 0.40 
        
        if task_context.time_sensitive:
            if capabilities.tokens_per_second > 300: 
                score += 0.20
            elif capabilities.tokens_per_second > 100:
                score += 0.10
            else:
                score -= 0.15
        
        if task_context.estimated_complexity.value >= 4: 
            if capabilities.reasoning_capability > 0.9:
                score += 0.15
            elif capabilities.reasoning_capability > 0.8:
                score += 0.10
        else:  
            if capabilities.reasoning_capability < 0.8: 
                score += 0.05
                
        if task_context.context_size > 80000 and capabilities.name == 'gemini-1.5-pro':
            score += 0.30
        
        if (task_context.time_sensitive and 
            task_context.estimated_complexity.value <= 2 and
            capabilities.name == 'llama-3.1-8b-instant'):
            score += 0.40
        
        if (task_context.task_type == TaskType.DEBUGGING and
            capabilities.name == 'deepseek-coder'):
            score += 0.25
        
        if (task_context.estimated_complexity.value <= 2 and
            capabilities.name == 'claude-3.5-sonnet'):
            score -= 0.25
        
        if task_context.has_multimodal_input:
            if capabilities.multimodal:
                score += 0.15
            else:
                score -= 0.50  
        
        if any(lang in capabilities.code_languages_strength 
               for lang in task_context.languages):
            score += 0.05
        
        if user_preferences:
            if user_preferences.get('preferred_llm') == capabilities.name:
                score += 0.15
            elif user_preferences.get('avoid_llm') == capabilities.name:
                score -= 0.30
        
        return max(0.0, min(1.0, score))
    
    def route_request(self, task_context: TaskContext, 
                     user_preferences: Optional[Dict] = None) -> Tuple[str, float]:
        """Route request using rule-based logic"""
        scores = {}
        for llm_name, capabilities in self.llm_registry.items():
            score = self.calculate_llm_score(capabilities, task_context, user_preferences)
            scores[llm_name] = score
        
        best_llm = max(scores.items(), key=lambda x: x[1])
        return best_llm[0], best_llm[1]

class HybridIntelligentRouter:
    """Hybrid router combining rule-based intelligence with contextual bandits"""
    
    def __init__(self, state_file: str = "router_state.json"):
        self.state_file = state_file
        self.llm_registry = self._initialize_llm_registry()
        self.task_analyzer = TaskAnalyzer()
        
        self.rule_based_router = RuleBasedRouter(self.llm_registry)
        self.bandit_router = ContextualBandit(
            list(self.llm_registry.keys()), 
            context_dim=14,
            alpha=0.5
        )
        
        self.bandit_router.load_state(self.state_file)
        
        self.bandit_confidence_threshold = 0.6
        self.min_bandit_interactions = 10
        
        self.performance_history = []
        
    def _initialize_llm_registry(self) -> Dict[str, LLMCapabilities]:
        """Initialize enhanced LLM capabilities registry"""
        return {
            'claude-3.5-sonnet': LLMCapabilities(
                name='claude-3.5-sonnet',
                max_context=200000,
                tokens_per_second=50.0,
                cost_per_1k_input=3.0,
                cost_per_1k_output=15.0,
                strengths=[TaskType.CODE_GENERATION, TaskType.REFACTORING, 
                          TaskType.ARCHITECTURE, TaskType.ANALYSIS],
                code_languages_strength=['python', 'javascript', 'typescript', 'rust', 'go'],
                reasoning_capability=0.95,
                latency_ms=1200,
                reliability_score=0.95,
                multimodal=True
            ),
            'gpt-4-turbo': LLMCapabilities(
                name='gpt-4-turbo',
                max_context=128000,
                tokens_per_second=40.0,
                cost_per_1k_input=10.0,
                cost_per_1k_output=30.0,
                strengths=[TaskType.CODE_GENERATION, TaskType.DOCUMENTATION, 
                          TaskType.ARCHITECTURE],
                code_languages_strength=['python', 'javascript', 'java', 'c++'],
                reasoning_capability=0.90,
                latency_ms=1500,
                reliability_score=0.92,
                multimodal=True
            ),
            'llama-3.1-8b-instant': LLMCapabilities(
                name='llama-3.1-8b-instant',
                max_context=8192,
                tokens_per_second=500.0,  # Very fast
                cost_per_1k_input=0.59,
                cost_per_1k_output=0.79,
                strengths=[TaskType.CODE_COMPLETION, TaskType.CODE_GENERATION, TaskType.DEBUGGING],
                code_languages_strength=['python', 'javascript'],
                reasoning_capability=0.75,
                latency_ms=200,
                reliability_score=0.88
            ),
            'deepseek-coder': LLMCapabilities(
                name='deepseek-coder',
                max_context=16384,
                tokens_per_second=100.0,
                cost_per_1k_input=0.14,
                cost_per_1k_output=0.28,
                strengths=[TaskType.CODE_GENERATION, TaskType.DEBUGGING, 
                          TaskType.REFACTORING],
                code_languages_strength=['python', 'java', 'cpp', 'javascript', 'go'],
                reasoning_capability=0.80,
                latency_ms=800,
                reliability_score=0.90
            ),
            'gemini-1.5-pro': LLMCapabilities(
                name='gemini-1.5-pro',
                max_context=1000000, 
                tokens_per_second=35.0,
                cost_per_1k_input=1.25,
                cost_per_1k_output=5.0,
                strengths=[TaskType.ANALYSIS, TaskType.DOCUMENTATION, 
                          TaskType.MULTIMODAL, TaskType.CODE_GENERATION],
                code_languages_strength=['python', 'javascript', 'java'],
                reasoning_capability=0.85,
                latency_ms=1800,
                reliability_score=0.89,
                multimodal=True
            )
        }
    
    def route_request(self, user_input: str, context_info: Dict, 
                     user_preferences: Optional[Dict] = None) -> RoutingDecision:
        """Main routing method using hybrid approach"""
        # Analyze task
        task_context = self.task_analyzer.analyze_task(user_input, context_info)
        context_vector = self.bandit_router._compute_context_vector(task_context)
        
        rule_llm, rule_confidence = self.rule_based_router.route_request(
            task_context, user_preferences
        )
        
        bandit_llm, bandit_confidence = None, None
        if self.bandit_router.has_sufficient_data(task_context, self.min_bandit_interactions):
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
        """Generate human-readable reasoning for rule-based decisions"""
        capabilities = self.llm_registry[selected_llm]
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
        
        if not reasons:
            reasons.append("Best overall match for task requirements")
        
        return f"Selected {selected_llm}: " + ", ".join(reasons)
    
    def record_performance(self, llm_name: str, task_context: TaskContext, 
                          context_vector: List[float], success: bool, 
                          response_time: float, cost: float, 
                          quality_score: float, user_satisfaction: float):
        """Record performance for both tracking and bandit learning"""
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
