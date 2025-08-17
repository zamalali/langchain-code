"""
Comprehensive test suite for the Hybrid Intelligent LLM Router
Tests various scenarios to validate routing decisions and learning performance
"""

import json
import time
import random
from typing import Dict, List
from router import HybridIntelligentRouter, TaskType, RoutingDecision

class HybridRouterTestSuite:
    def __init__(self):
        self.router = HybridIntelligentRouter()
        self.test_results = []
        
    def run_comprehensive_tests(self):
        """Run all test scenarios and analyze results"""
        print(" Starting Comprehensive Hybrid Router Tests\n")
        
        test_scenarios = self._get_test_scenarios()
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"Test {i:2d}: {scenario['description']}")
            print(f"Input: '{scenario['input']}'")
            
            # Run the router
            start_time = time.time()
            decision = self.router.route_request(
                scenario['input'], 
                scenario['context']
            )
            routing_time = time.time() - start_time
            
            # Analyze the task context for verification
            task_context = self.router.task_analyzer.analyze_task(
                scenario['input'], 
                scenario['context']
            )
            
            # Store results
            result = {
                'scenario': scenario,
                'selected_llm': decision.selected_llm,
                'confidence': decision.confidence,
                'task_type': task_context.task_type.value,
                'complexity': task_context.estimated_complexity.name,
                'routing_time_ms': routing_time * 1000,
                'reasoning': decision.reasoning,
                'decision_source': decision.decision_source,
                'rule_based_score': decision.rule_based_score,
                'bandit_score': decision.bandit_score,
                'expected_llm': scenario.get('expected_llm'),
                'expected_task_type': scenario.get('expected_task_type')
            }
            self.test_results.append(result)
            
            # Display results
            print(f"Selected: {decision.selected_llm} (confidence: {decision.confidence:.3f})")
            print(f"Source: {decision.decision_source}")
            print(f"Task Type: {task_context.task_type.value}")
            print(f"Complexity: {task_context.estimated_complexity.name}")
            print(f"Routing Time: {routing_time*1000:.1f}ms")
            print(f"Reasoning: {decision.reasoning}")
            
            # Show rule vs bandit scores
            if decision.bandit_score is not None:
                print(f"Rule Score: {decision.rule_based_score:.3f}, Bandit Score: {decision.bandit_score:.3f}")
            
            # Validation check
            if scenario.get('expected_llm') and decision.selected_llm != scenario['expected_llm']:
                print(f"⚠️  Expected {scenario['expected_llm']}, got {decision.selected_llm}")
            else:
                print("✅ Routing decision looks appropriate")
            
            # Simulate performance feedback (for bandit learning)
            self._simulate_performance_feedback(decision, task_context, scenario)
            
            print("-" * 80)
        
        self._analyze_results()
    
    def _simulate_performance_feedback(self, decision: RoutingDecision, 
                                     task_context, scenario: Dict):
        """Simulate performance feedback to train the bandit"""
        # Simulate realistic performance metrics
        llm_performance_profiles = {
            'claude-3.5-sonnet': {'success': 0.95, 'quality': 0.9, 'speed': 0.7},
            'gemini-1.5-pro': {'success': 0.90, 'quality': 0.85, 'speed': 0.6},
            'groq-llama3-70b': {'success': 0.85, 'quality': 0.75, 'speed': 0.95},
            'deepseek-coder': {'success': 0.88, 'quality': 0.82, 'speed': 0.8},
            'gpt-4-turbo': {'success': 0.92, 'quality': 0.88, 'speed': 0.65}
        }
        
        profile = llm_performance_profiles.get(decision.selected_llm, 
                                             {'success': 0.8, 'quality': 0.75, 'speed': 0.7})
        
        # Add some task-specific adjustments
        quality_bonus = 0.0
        speed_bonus = 0.0
        
        # Complexity adjustments
        if task_context.estimated_complexity.value >= 4 and decision.selected_llm == 'claude-3.5-sonnet':
            quality_bonus += 0.1  # Claude excels at complex tasks
        
        # Speed adjustments
        if task_context.time_sensitive and decision.selected_llm == 'groq-llama3-70b':
            speed_bonus += 0.1  # Groq excels at speed
        
        # Context size adjustments
        if task_context.context_size > 50000 and decision.selected_llm == 'gemini-1.5-pro':
            quality_bonus += 0.15  # Gemini handles large context well
        
        # Multimodal adjustments
        if task_context.has_multimodal_input and decision.selected_llm in ['claude-3.5-sonnet', 'gemini-1.5-pro']:
            quality_bonus += 0.1
        
        # Generate realistic metrics with some randomness
        success = random.random() < (profile['success'] + 0.1 * quality_bonus)
        quality_score = min(1.0, profile['quality'] + quality_bonus + random.gauss(0, 0.05))
        response_time = max(0.1, 2.0 / (profile['speed'] + speed_bonus) + random.gauss(0, 0.3))
        user_satisfaction = min(1.0, (quality_score + (1.0 if success else 0.0)) / 2 + random.gauss(0, 0.1))
        
        # Estimate cost (simplified)
        context_tokens = task_context.context_size + 500  # Estimated output
        llm_caps = self.router.llm_registry[decision.selected_llm]
        cost = (context_tokens * llm_caps.cost_per_1k_input / 1000 + 
                500 * llm_caps.cost_per_1k_output / 1000)
        
        # Record performance for bandit learning
        self.router.record_performance(
            llm_name=decision.selected_llm,
            task_context=task_context,
            context_vector=decision.context_vector,
            success=success,
            response_time=response_time,
            cost=cost,
            quality_score=max(0, quality_score),
            user_satisfaction=max(0, user_satisfaction)
        )
    
    def _get_test_scenarios(self) -> List[Dict]:
        """Define comprehensive test scenarios covering various use cases"""
        return [
            # 1. Simple/Quick Tasks
            {
                'description': 'Simple syntax fix',
                'input': 'fix this quick typo in my function',
                'context': {
                    'context_size': 50,
                    'file_count': 1,
                    'languages': ['python']
                },
                'expected_llm': 'groq-llama3-70b',
                'expected_task_type': TaskType.DEBUGGING
            },
            
            # 2. Code Completion
            {
                'description': 'Code completion task',
                'input': 'complete this function for me, just need the basic logic',
                'context': {
                    'context_size': 150,
                    'file_count': 1,
                    'languages': ['javascript']
                },
                'expected_task_type': TaskType.CODE_COMPLETION
            },
            
            # 3. Complex Architecture
            {
                'description': 'Complex architecture redesign',
                'input': 'redesign the entire microservices architecture to improve scalability',
                'context': {
                    'context_size': 15000,
                    'file_count': 50,
                    'languages': ['python', 'docker', 'kubernetes']
                },
                'expected_llm': 'claude-3.5-sonnet',
                'expected_task_type': TaskType.ARCHITECTURE
            },
            
            # 4. Large Codebase Analysis - Should favor Gemini
            {
                'description': 'Comprehensive codebase analysis',
                'input': 'analyze my entire codebase and identify performance bottlenecks',
                'context': {
                    'context_size': 100000,
                    'file_count': 200,
                    'languages': ['python', 'javascript', 'sql']
                },
                'expected_llm': 'gemini-1.5-pro',
                'expected_task_type': TaskType.ANALYSIS
            },
            
            # 5. Documentation Generation - Large context
            {
                'description': 'Extensive documentation task',
                'input': 'create comprehensive documentation for this entire project with best practices',
                'context': {
                    'context_size': 85000,
                    'file_count': 120,
                    'languages': ['python', 'markdown']
                },
                'expected_llm': 'gemini-1.5-pro',
                'expected_task_type': TaskType.DOCUMENTATION
            },
            
            # 6. Multimodal UI Task
            {
                'description': 'UI design from image',
                'input': 'look at this screenshot and build a similar interface using React',
                'context': {
                    'context_size': 2000,
                    'file_count': 5,
                    'languages': ['react', 'css'],
                    'has_image': True
                },
                'expected_task_type': TaskType.MULTIMODAL
            },
            
            # 7. Urgent/Time-sensitive - Should favor Groq
            {
                'description': 'Urgent bug fix',
                'input': 'urgent: production is down, need to fix this database connection issue asap',
                'context': {
                    'context_size': 300,
                    'file_count': 2,
                    'languages': ['python', 'sql']
                },
                'expected_llm': 'groq-llama3-70b',
                'expected_task_type': TaskType.DEBUGGING
            },
            
            # 8. Code Refactoring
            {
                'description': 'Code refactoring',
                'input': 'refactor this legacy code to use modern design patterns',
                'context': {
                    'context_size': 5000,
                    'file_count': 10,
                    'languages': ['java']
                },
                'expected_task_type': TaskType.REFACTORING
            },
            
            # 9. Test Generation
            {
                'description': 'Test suite creation',
                'input': 'generate comprehensive unit tests for my API endpoints',
                'context': {
                    'context_size': 3000,
                    'file_count': 8,
                    'languages': ['python', 'pytest']
                },
                'expected_task_type': TaskType.TESTING
            },
            
            # 10. Algorithm Implementation - Complex
            {
                'description': 'Complex algorithm',
                'input': 'implement an advanced machine learning algorithm for recommendation system',
                'context': {
                    'context_size': 1000,
                    'file_count': 3,
                    'languages': ['python']
                },
                'expected_llm': 'claude-3.5-sonnet',
                'expected_task_type': TaskType.CODE_GENERATION
            },
            
            # 11. Simple Code Generation
            {
                'description': 'Basic function creation',
                'input': 'write a simple function to validate email addresses',
                'context': {
                    'context_size': 100,
                    'file_count': 1,
                    'languages': ['python']
                },
                'expected_task_type': TaskType.CODE_GENERATION
            },
            
            # 12. Performance Optimization
            {
                'description': 'Performance optimization',
                'input': 'optimize performance of this data processing pipeline',
                'context': {
                    'context_size': 4000,
                    'file_count': 6,
                    'languages': ['python', 'pandas']
                },
                'expected_task_type': TaskType.CODE_GENERATION
            },
            
            # 13. Code Review
            {
                'description': 'Code review request',
                'input': 'review my code and suggest improvements for better maintainability',
                'context': {
                    'context_size': 2500,
                    'file_count': 4,
                    'languages': ['typescript']
                },
                'expected_task_type': TaskType.ANALYSIS
            },
            
            # 14. Database Design
            {
                'description': 'Database schema design',
                'input': 'design a database schema for e-commerce platform',
                'context': {
                    'context_size': 800,
                    'file_count': 2,
                    'languages': ['sql']
                },
                'expected_task_type': TaskType.ARCHITECTURE
            },
            
            # 15. Migration Task
            {
                'description': 'Framework migration',
                'input': 'migrate this Flask app to FastAPI framework',
                'context': {
                    'context_size': 6000,
                    'file_count': 15,
                    'languages': ['python']
                },
                'expected_task_type': TaskType.REFACTORING
            },
            
            # 16. API Documentation
            {
                'description': 'API documentation',
                'input': 'create OpenAPI documentation for my REST endpoints',
                'context': {
                    'context_size': 1500,
                    'file_count': 5,
                    'languages': ['python', 'yaml']
                },
                'expected_task_type': TaskType.DOCUMENTATION
            },
            
            # 17. React Component
            {
                'description': 'React component creation',
                'input': 'build a reusable data visualization component in React',
                'context': {
                    'context_size': 800,
                    'file_count': 3,
                    'languages': ['react', 'javascript']
                },
                'expected_task_type': TaskType.CODE_GENERATION
            },
            
            # 18. Security Audit - Large codebase
            {
                'description': 'Security analysis',
                'input': 'audit my entire authentication system for security vulnerabilities',
                'context': {
                    'context_size': 45000,
                    'file_count': 25,
                    'languages': ['python', 'javascript']
                },
                'expected_task_type': TaskType.ANALYSIS
            },
            
            # 19. Configuration Setup
            {
                'description': 'Environment configuration',
                'input': 'set up Docker configuration for my development environment',
                'context': {
                    'context_size': 400,
                    'file_count': 2,
                    'languages': ['dockerfile', 'yaml']
                },
                'expected_task_type': TaskType.CODE_GENERATION
            },
            
            # 20. Large Refactoring - Massive codebase
            {
                'description': 'Major codebase refactoring',
                'input': 'restructure entire project to follow clean architecture principles',
                'context': {
                    'context_size': 150000,
                    'file_count': 300,
                    'languages': ['python', 'javascript']
                },
                'expected_llm': 'gemini-1.5-pro',
                'expected_task_type': TaskType.ARCHITECTURE
            },
            
            # 21. Visual Design Implementation
            {
                'description': 'Design to code conversion',
                'input': 'convert this Figma design to working HTML/CSS code',
                'context': {
                    'context_size': 1200,
                    'file_count': 4,
                    'languages': ['html', 'css', 'javascript'],
                    'has_image': True
                },
                'expected_task_type': TaskType.MULTIMODAL
            },
            
            # 22. Integration Testing
            {
                'description': 'Integration test setup',
                'input': 'create integration tests for microservices communication',
                'context': {
                    'context_size': 4500,
                    'file_count': 12,
                    'languages': ['python', 'docker']
                },
                'expected_task_type': TaskType.TESTING
            },
            
            # 23. Quick debugging task
            {
                'description': 'Quick error fix',
                'input': 'fix this error in my Python script quickly',
                'context': {
                    'context_size': 80,
                    'file_count': 1,
                    'languages': ['python']
                },
                'expected_llm': 'groq-llama3-70b',
                'expected_task_type': TaskType.DEBUGGING
            },
            
            # 24. Massive documentation project
            {
                'description': 'Enterprise documentation',
                'input': 'go through the entire codebase and create documentation extensively just like how big codebases are documented with best practices',
                'context': {
                    'context_size': 250000,
                    'file_count': 500,
                    'languages': ['python', 'javascript', 'java']
                },
                'expected_llm': 'gemini-1.5-pro',
                'expected_task_type': TaskType.DOCUMENTATION
            }
        ]
    
    def _analyze_results(self):
        """Analyze test results and provide comprehensive summary"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE HYBRID ROUTING ANALYSIS")
        print("="*80)
        
        # LLM Selection Distribution
        llm_counts = {}
        for result in self.test_results:
            llm = result['selected_llm']
            llm_counts[llm] = llm_counts.get(llm, 0) + 1
        
        print("\n🎯 LLM Selection Distribution:")
        for llm, count in sorted(llm_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.test_results)) * 100
            print(f"  {llm:20s}: {count:2d} tasks ({percentage:5.1f}%)")
        
        # Decision Source Analysis
        source_counts = {}
        for result in self.test_results:
            source = result['decision_source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print("\n🧠 Decision Source Distribution:")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.test_results)) * 100
            print(f"  {source:15s}: {count:2d} decisions ({percentage:5.1f}%)")
        
        # Task Type Distribution
        task_type_counts = {}
        for result in self.test_results:
            task_type = result['task_type']
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
        
        print("\n🏷️  Task Type Classification:")
        for task_type, count in sorted(task_type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.test_results)) * 100
            print(f"  {task_type:20s}: {count:2d} tasks ({percentage:5.1f}%)")
        
        # Complexity Distribution
        complexity_counts = {}
        for result in self.test_results:
            complexity = result['complexity']
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        print("\n⚡ Task Complexity Distribution:")
        for complexity, count in sorted(complexity_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.test_results)) * 100
            print(f"  {complexity:12s}: {count:2d} tasks ({percentage:5.1f}%)")
        
        # Performance Metrics
        avg_confidence = sum(r['confidence'] for r in self.test_results) / len(self.test_results)
        avg_routing_time = sum(r['routing_time_ms'] for r in self.test_results) / len(self.test_results)
        
        print(f"\n⏱️  Performance Metrics:")
        print(f"  Average Confidence Score: {avg_confidence:.3f}")
        print(f"  Average Routing Time:     {avg_routing_time:.1f}ms")
        print(f"  Total Tests Executed:     {len(self.test_results)}")
        
        high_confidence = len([r for r in self.test_results if r['confidence'] > 0.8])
        medium_confidence = len([r for r in self.test_results if 0.6 <= r['confidence'] <= 0.8])
        low_confidence = len([r for r in self.test_results if r['confidence'] < 0.6])
        
        print(f"\n📈 Confidence Level Distribution:")
        print(f"  High Confidence (>0.8):     {high_confidence:2d} tasks ({high_confidence/len(self.test_results)*100:5.1f}%)")
        print(f"  Medium Confidence (0.6-0.8): {medium_confidence:2d} tasks ({medium_confidence/len(self.test_results)*100:5.1f}%)")
        print(f"  Low Confidence (<0.6):       {low_confidence:2d} tasks ({low_confidence/len(self.test_results)*100:5.1f}%)")
        
        large_context_tasks = [r for r in self.test_results if r['scenario']['context']['context_size'] > 50000]
        urgent_tasks = [r for r in self.test_results if 'urgent' in r['scenario']['input'].lower() or 'quick' in r['scenario']['input'].lower() or 'asap' in r['scenario']['input'].lower()]
        multimodal_tasks = [r for r in self.test_results if r['scenario']['context'].get('has_image', False)]
        
        print(f"\n🎯 Specialized Routing Analysis:")
        
        if large_context_tasks:
            gemini_for_large = len([r for r in large_context_tasks if 'gemini' in r['selected_llm']])
            claude_for_large = len([r for r in large_context_tasks if 'claude' in r['selected_llm']])
            print(f"  Large Context Tasks ({len(large_context_tasks)}): Gemini {gemini_for_large}, Claude {claude_for_large}")
        
        if urgent_tasks:
            groq_for_urgent = len([r for r in urgent_tasks if 'groq' in r['selected_llm']])
            print(f"  Urgent Tasks ({len(urgent_tasks)}): Groq selected {groq_for_urgent}/{len(urgent_tasks)} times")
        
        if multimodal_tasks:
            multimodal_capable = len([r for r in multimodal_tasks if r['selected_llm'] in ['claude-3.5-sonnet', 'gemini-1.5-pro', 'gpt-4-turbo']])
            print(f"  Multimodal Tasks ({len(multimodal_tasks)}): Multimodal-capable LLMs selected {multimodal_capable}/{len(multimodal_tasks)} times")
        
        # Bandit Learning Progress
        bandit_decisions = [r for r in self.test_results if r['decision_source'] == 'bandit']
        print(f"\n🤖 Bandit Learning Progress:")
        print(f"  Bandit Decisions: {len(bandit_decisions)}/{len(self.test_results)} ({len(bandit_decisions)/len(self.test_results)*100:.1f}%)")
        print(f"  Total Interactions: {self.router.bandit_router.total_interactions}")
        print(f"  Learning Status: {'Active' if len(bandit_decisions) > 0 else 'Collecting Data'}")
        
        # Expected vs Actual Analysis
        expectation_matches = 0
        total_expectations = 0
        for result in self.test_results:
            if result.get('expected_llm'):
                total_expectations += 1
                if result['selected_llm'] == result['expected_llm']:
                    expectation_matches += 1
        
        if total_expectations > 0:
            match_rate = expectation_matches / total_expectations * 100
            print(f"\n✅ Expectation Matching:")
            print(f"  Expected LLM matches: {expectation_matches}/{total_expectations} ({match_rate:.1f}%)")
        
        most_confident = max(self.test_results, key=lambda x: x['confidence'])
        print(f"\n🧠 Key Insights:")
        print(f"  Most confident: {most_confident['selected_llm']} for '{most_confident['scenario']['description']}' (confidence: {most_confident['confidence']:.3f})")
        
        fastest = min(self.test_results, key=lambda x: x['routing_time_ms'])
        print(f"  Fastest routing: {fastest['routing_time_ms']:.1f}ms for '{fastest['scenario']['description']}'")
        
        self._save_results_to_file()
        
        print(f"\n✅ Hybrid router testing complete! Results saved to 'hybrid_router_test_results.json'")
        print(f"📊 Router state saved to '{self.router.state_file}'")
        print("="*80)
    
    def _save_results_to_file(self):
        """Save detailed results to JSON file for further analysis"""
        # Prepare results for JSON serialization
        json_results = []
        for result in self.test_results:
            json_result = result.copy()
            # Convert any enum values to strings
            if 'expected_task_type' in json_result and json_result['expected_task_type']:
                json_result['expected_task_type'] = json_result['expected_task_type'].value if hasattr(json_result['expected_task_type'], 'value') else str(json_result['expected_task_type'])
            json_results.append(json_result)
        
        with open('hybrid_router_test_results.json', 'w') as f:
            json.dump(json_results, f, indent=2, default=str)

def main():
    """Run the comprehensive hybrid router test suite"""
    test_suite = HybridRouterTestSuite()
    test_suite.run_comprehensive_tests()

if __name__ == "__main__":
    main()