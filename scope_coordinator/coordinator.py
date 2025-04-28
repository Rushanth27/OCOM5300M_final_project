from matplotlib import pyplot as plt
from safetensors.torch import load_model

from datetime import datetime
from matplotlib import pyplot as plt
from safetensors.torch import load_model
from typing import Dict, List, Union

import json
import os
import pandas as pd
import numpy as np

from scope_coordinator.agents.answer_agent import AnswerAgent
from scope_coordinator.agents.clarification_agent import ClarificationAgent
from scope_coordinator.agents.document_agent import DocumentAgent
from scope_coordinator.agents.risk_identification_agent import RiskIdentificationAgent
from scope_coordinator.client import BaseLLMClient, OpenAIClient, HuggingFaceClient
from scope_coordinator.models.risk_analyzer_gcn import risk_classifier

# Constants
RISK_THRESHOLD = 3  # Only process risks with scores above this threshold

class ScopingCoordinator:
    """Coordinates the scoping process between different agents."""

    def __init__(self, 
             answer_llm: BaseLLMClient,
             clarification_llm: BaseLLMClient,
             risk_identification_llm: BaseLLMClient,
             document_llm: BaseLLMClient,
             auto_answer: bool,
             max_risks: int,
             max_questions_per_risk: int,
             max_iterations: int,
             agent_settings: Dict):

        self.clarification_agent = ClarificationAgent(clarification_llm)
        self.document_agent = DocumentAgent(document_llm)
        self.risk_identification_agent = RiskIdentificationAgent(risk_identification_llm)
        self.answer_agent = AnswerAgent(answer_llm)

        self.max_iterations = max_iterations
        self.auto_answer = auto_answer
        self.max_risks = max_risks
        self.max_questions_per_risk = max_questions_per_risk

        self.iteration_history = []
        self.risk_model = None
        self.agent_settings = agent_settings

    def map_to_level(self, value: int) -> str:
        """
        Maps a numerical value from 1-5 to a risk level.

        Args:
            value (int): Integer value from 1-5

        Returns:
            str: Risk level category
        """
        risk_levels = {
            1: "Very Low",
            2: "Low",
            3: "Medium",
            4: "High",
            5: "Very High"
        }
        return risk_levels.get(value, "Unknown")
        
    def predict_risk(self, risk_description: str) -> Dict:
        """
        Predict risk using the GCN model.

        Args:
            risk_description: The description of the risk to analyze

        Returns:
            Dict containing impact, likelihood, and risk score
        """
        try:
            impact_model_path = "scope_coordinator/models/saved_models/gcn_model_impact/"
            likelihood_model_path = "scope_coordinator/models/saved_models/gcn_model_likelihood/"

            perdicted_impact = risk_classifier(risk_description, impact_model_path)
            perdicted_likelihood = risk_classifier(risk_description, likelihood_model_path)

            return {
                'risk_description': risk_description,
                'impact': {
                    'level': self.map_to_level(perdicted_impact['value']),
                    'value': perdicted_impact['value'],
                    'confidence': perdicted_impact['confidence']
                },
                'likelihood': {
                    'level': self.map_to_level(perdicted_likelihood['value']),
                    'value': perdicted_likelihood['value'],
                    'confidence': perdicted_likelihood['confidence']
                },
                'risk_score': perdicted_impact['value'] * perdicted_likelihood['value']
            }
        except Exception as e:
            print(f"Error in risk prediction: {str(e)}")
            # Return default medium risk if prediction fails
        

    def display_risk_analysis(self, risks: Dict):
        """
        Display the risk analysis results in a formatted way.

        Args:
            risks: Dictionary containing risk analysis results
        """
        print("\n=== Risk Analysis ===")
        print(f"Category: {risks.get('category', 'N/A')}")
        print(f"Risk Description: {risks['risk_description']}")
        print(f"Impact: {risks['impact']['level']} ({risks['impact']['value']})")
        print(f"Likelihood: {risks['likelihood']['level']} ({risks['likelihood']['value']})")
        print(f"Risk Score: {risks['risk_score']}")
        print(f"Consequences: {risks.get('consequences', 'N/A')}")

    def create_final_response(self, current_scope: str, all_risks: List[Dict], iteration_history: List[Dict]) -> Dict:
        """Create the final response dictionary."""
        final_iter = iteration_history[-1] if iteration_history else {
            "scope": current_scope,
            "risks": all_risks,
            "updated_scope": current_scope
        }

        return {
            "status": "completed",
            "iterations": len(iteration_history),
            "initial_scope": current_scope,
            "final_scope": final_iter["updated_scope"],
            "risks": all_risks,
            # the below is nice for user to view, excluding as not needed for testing
            # "final_document": self.document_agent.generate_document({
            #     "initial_request": current_scope,
            #     "iteration_history": iteration_history,
            #     "final_scope": final_iter["updated_scope"],
            #     "final_risks": all_risks
            # })
        }

    def process_request(self, initial_request: Union[str, List[Dict]]) -> Dict:
        """Process a scoping request through multiple iterations."""
        try:
            # Initial setup of scope from input
            if isinstance(initial_request, str):
                current_scope = initial_request
                messages = [{
                    "role": "user",
                    "content": [{"type": "text", "text": initial_request}]
                }]
            else:
                messages = initial_request
                current_scope = " ".join(
                    item["content"][0]["text"]
                    for item in messages
                    if isinstance(item, dict) and "content" in item
                )

            all_risk_assessments = []

            # Main iteration loop
            for iteration in range(self.max_iterations):
                print(f"\n{'#' * 50}")
                print(f"Iteration {iteration + 1}")
                print(f"{'#' * 50}")
                print(f"\nCurrent Scope:")
                print(f"{'=' * 50}")
                print(current_scope)
                print(f"{'=' * 50}\n")

                # Step 1: Identify risks from current scope
                identified_risks = self.risk_identification_agent.identify_risks(current_scope)
                self.risk_identification_agent.display_identified_risks(identified_risks)

                # Step 2: Analyze and assess risks
                high_risks = []
                iteration_risk_assessments = []
                print("\n=== Risk Assessments ===")

                for risk in identified_risks:
                    risk_assessment = self.predict_risk(risk['description'])

                    risk_assessment['category'] = risk['category']
                    risk_assessment['consequences'] = risk['consequences']

                    # print("risk assessment:", risk_assessment)
                    
                    self.display_risk_analysis(risk_assessment)
                    iteration_risk_assessments.append(risk_assessment)

                    if risk_assessment['risk_score'] > RISK_THRESHOLD:
                        high_risks.append(risk_assessment)
                        print(f"Warning: Risk score {risk_assessment['risk_score']} exceeds threshold")
                    else:
                        print(f"Success: Risk score {risk_assessment['risk_score']} below threshold")

                all_risk_assessments = iteration_risk_assessments

                # If no high risks, end the iteration process
                if not high_risks:
                    print("\nNo high-risk items identified that require clarification.")
                    break

                # Step 3: Process high-risk items
                sorted_high_risks = sorted(high_risks, key=lambda x: x['risk_score'], reverse=True)

                print("\nRisks by score (highest to lowest):")
                for i, risk in enumerate(sorted_high_risks):
                    print(f"Risk {i + 1} | Score: {risk['risk_score']} | Risk Description: {risk['risk_description']}")

                top_high_risks = sorted_high_risks[:self.max_risks]
                print(f"\nProcessing top {len(top_high_risks)} highest risk items.")

                # Step 4: Generate and handle clarifying questions
                all_questions = []
                for risk_assessment in top_high_risks:
                    questions = self.clarification_agent.generate_questions(
                        risk_assessment["risk_description"]
                    )[:self.max_questions_per_risk]
                    all_questions.extend([
                        {
                            "risk_category": risk_assessment["category"],
                            "risk_description": risk_assessment["risk_description"],
                            "question": q
                        } for q in questions
                    ])

                if not all_questions:
                    print("\nNo further questions needed. Analysis complete.")
                    break

                # Step 5: Get answers (auto or manual)
                print("\nClarifying Questions for Top High-Risk Items:")
                qa_pairs = []
                for i, q_data in enumerate(all_questions, 1):
                    print(f"\nRegarding the risk: {q_data['risk_description'][:100]}...")
                    print(f"Q{i}: {q_data['question']}")

                    if self.auto_answer:
                        answer = self.answer_agent.generate_answer(
                            question=q_data['question'],
                            risk_description=q_data['risk_description'],
                            risk_category=q_data['risk_category']
                        )
                        print(f"Auto-generated answer: {answer}")
                    else:
                        answer = input("Your answer: ")

                    qa_pairs.append({
                        "risk_category": q_data['risk_category'],
                        "risk_description": q_data['risk_description'],
                        "question": q_data['question'],
                        "answer": answer
                    })

                # Step 6: Update scope based on answers
                updated_scope = self.document_agent.update_scope(current_scope, qa_pairs)

                print(f"\n{'=' * 50}")
                print("Updated Scope:")
                print(f"{'=' * 50}")
                print(updated_scope)
                print(f"{'=' * 50}\n")

                # Step 7: Calculate similarity metrics
                #calculate similarity between risks in the iteration
                risk_similarity_metrics = self.calculate_all_risk_similarities(all_risk_assessments)
                
                # Calculate similarity metrics
                scope_similarity_metrics = self.calculate_scope_similarity(current_scope, updated_scope)

                # Calculate Jaccard similarity
                jaccard_similarity = self.jaccard_similarity(current_scope, updated_scope)
                scope_similarity_metrics.update(jaccard_similarity)

                # Print the metrics
                print(f"\n{'=' * 50}")
                print(f"Scope Similarity Metrics (Iteration {iteration + 1}):")
                print(f"{'=' * 50}")
                print(f"Cosine Similarity: {scope_similarity_metrics['similarity_score']:.4f}")
                print(f"Cosine Difference: {scope_similarity_metrics['semantic_difference']:.4f}")
                print(f"Jaccard Similarity: {scope_similarity_metrics['jaccard_similarity']:.4f}")
                print(f"{'=' * 50}\n")

                # Step 8: Store iteration history
                # Store iteration history
                self.iteration_history.append({
                    "iteration": iteration + 1,
                    "scope": current_scope,
                    "messages": messages.copy(),
                    "risks": iteration_risk_assessments,
                    "qa_pairs": qa_pairs,
                    "updated_scope": updated_scope,
                    "scope_similarity_metrics": scope_similarity_metrics,
                    "risk_similarity_metrics": risk_similarity_metrics
                })

                print("Iteration history:", self.iteration_history)

                # Check if scope has stabilized
                if current_scope == updated_scope:
                    print("\nScope has stabilized. Analysis complete.")
                    break

                current_scope = updated_scope

            if self.iteration_history:
                print("\n=== Similarity Metrics Per Iteration ===")
                print(f"{'Iteration':<10} {'Similarity Score':<20} {'Semantic Difference':<20}")
                print("-" * 50)
                for item in self.iteration_history:
                    print(f"{item['iteration']:<10} "
                          f"{item['scope_similarity_metrics']['similarity_score']:<20.4f} "
                          f"{item['scope_similarity_metrics']['semantic_difference']:<20.4f}")
                print("-" * 50)
                
                self.plot_and_store_metrics(plot_now = True)

            return self.create_final_response(
                self.iteration_history[0]["scope"],
                all_risk_assessments,
                self.iteration_history
            )

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def calculate_scope_similarity(self, previous_scope: str, updated_scope: str) -> Dict:
        """
        Calculate semantic similarity between two scope versions.

        Args:
            previous_scope: The original scope text
            updated_scope: The updated scope text

        Returns:
            Dict containing similarity metrics
        """

        # Lazy-load the model only when needed
        if not hasattr(self, 'similarity_model'):
            from sentence_transformers import SentenceTransformer, util
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Calculate embeddings
        embedding_previous = self.similarity_model.encode(previous_scope, convert_to_tensor=True)
        embedding_updated = self.similarity_model.encode(updated_scope, convert_to_tensor=True)

        # Calculate similarity metrics
        from sentence_transformers import util
        cosine_score = util.cos_sim(embedding_previous, embedding_updated).item()
        semantic_difference = 1 - cosine_score

        return {
            "similarity_score": cosine_score,
            "semantic_difference": semantic_difference
        }

    def jaccard_similarity(self, previous_scope: str, updated_scope: str) -> Dict:
        """
        Calculate Jaccard similarity between two scope versions.

        Args:
            previous_scope: The original scope text
            updated_scope: The updated scope text

        Returns:
            Dict containing Jaccard similarity
        """
        a = set(previous_scope.split())
        b = set(updated_scope.split())
        jaccard_similarity = len(a.intersection(b)) / len(a.union(b))
        return {
            "jaccard_similarity": jaccard_similarity
        }

    def calculate_risk_similarity(self, previous_risk: Dict, updated_risk: Dict) -> Dict:
        """
        Calculate semantic similarity between two risk versions.

        Args:
            previous_risk: The original risk dictionary
            updated_risk: The updated risk dictionary

        Returns:
            Dict containing similarity metrics
        """
        # Lazy-load the model only when needed
        if not hasattr(self, 'similarity_model'):
            from sentence_transformers import SentenceTransformer, util
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Calculate embeddings
        embedding_previous = self.similarity_model.encode(previous_risk['description'], convert_to_tensor=True)
        embedding_updated = self.similarity_model.encode(updated_risk['description'], convert_to_tensor=True)

        # Calculate similarity metrics
        from sentence_transformers import util
        cosine_score = util.cos_sim(embedding_previous, embedding_updated).item()
        semantic_difference = 1 - cosine_score

        return {
            "similarity_score": cosine_score,
            "semantic_difference": semantic_difference
        }

    def calculate_all_risk_similarities(self, risks: List[Dict], threshold: float = 0.7) -> Dict:
        """
        Calculate semantic similarity between all pairs of risks in a list,
        and compute quality metrics from the similarity matrix.
        
        Args:
            risks: List of risk dictionaries.
            threshold: Similarity threshold for counting highly similar risks.
            
        Returns:
            Dict containing similarity matrix and various summary statistics.
        """
        # Lazy-load the model only when needed
        if not hasattr(self, 'similarity_model'):
            from sentence_transformers import SentenceTransformer, util
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        n_risks = len(risks)
        descriptions = [risk['risk_description'] for risk in risks]
        
        # Calculate embeddings for all descriptions
        embeddings = self.similarity_model.encode(descriptions, convert_to_tensor=True)
        
        # Calculate similarity matrix
        from sentence_transformers import util
        similarity_matrix = util.cos_sim(embeddings, embeddings)
        
        similarity_matrix_list = similarity_matrix.tolist()  # For easier external use if needed
        
        # Calculate metrics
        similarity_values = []
        max_similarities = []
        
        for i in range(n_risks):
            row = similarity_matrix[i]
            # Ignore self-similarity at (i,i) = 1
            similarities_to_others = [row[j].item() for j in range(n_risks) if j != i]
            
            if similarities_to_others:
                max_sim = max(similarities_to_others)
                max_similarities.append(max_sim)
            
            # Collect upper triangle only (i < j) for average of all pairwise similarities
            for j in range(i+1, n_risks):
                similarity_values.append(similarity_matrix[i][j].item())
        
        # Summary statistics
        avg_similarity = sum(similarity_values) / len(similarity_values) if similarity_values else 0
        avg_max_similarity = sum(max_similarities) / len(max_similarities) if max_similarities else 0
        risk_drifts = [1 - sim for sim in max_similarities]
        avg_risk_drift = sum(risk_drifts) / len(risk_drifts) if risk_drifts else 0
        proportion_above_threshold = sum(1 for sim in max_similarities if sim > threshold) / len(max_similarities) if max_similarities else 0
        
        return {
            "similarity_matrix": similarity_matrix_list,
            "risk_descriptions": descriptions,
            "avg_similarity": avg_similarity,
            "max_similarity": max(similarity_values) if similarity_values else 0,
            "min_similarity": min(similarity_values) if similarity_values else 0,
            "avg_max_similarity": avg_max_similarity,
            "avg_risk_drift": avg_risk_drift,
            "proportion_above_threshold": proportion_above_threshold,
            "max_similarities": max_similarities,
            "risk_drifts": risk_drifts
        }


    def plot_and_store_metrics(self, plot_now = True, output_dir = "plots"):
        """
        Plot and store metrics from the iteration history in a timestamped subfolder.
        Args:
            output_dir (str): Base directory where output files will be stored
        """
        if not self.iteration_history:
            print("No iteration history to plot")
            return

        # Generate timestamp for the subfolder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create timestamped subfolder
        run_output_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(run_output_dir, exist_ok=True)

        last_iteration = self.iteration_history[-1]['iteration']

        # Save iteration history into newly created folder
        with open(os.path.join(run_output_dir, 'iteration_history.json'), 'w') as f:
            json.dump(self.iteration_history, f, indent=4)

        df = pd.DataFrame(self.iteration_history)
        df.to_csv(os.path.join(run_output_dir, 'iteration_history.csv'), index=False)
        
        # Extract impact and likelihood values from risks in each iteration
        impact_stats_per_iteration = []
        for item in self.iteration_history:
            try:
                impact_values = []
                impact_confidence_values = []
                likelihood_values = []
                likelihood_confidence_values = []

                for risk in item.get('risks', []):
                    try:
                        if isinstance(risk.get('impact', {}), dict):
                            impact_value = risk.get('impact', {}).get('value')
                            impact_confidence = risk.get('impact', {}).get('confidence')
                            if impact_value is not None:
                                impact_values.append(float(impact_value))
                            if impact_confidence is not None:
                                impact_confidence_values.append(float(impact_confidence))


                        if isinstance(risk.get('likelihood', {}), dict):
                            likelihood_value = risk.get('likelihood', {}).get('value')
                            likelihood_confidence = risk.get('likelihood', {}).get('confidence')
                            if likelihood_value is not None:
                                likelihood_values.append(float(likelihood_value))
                            if likelihood_confidence is not None:
                                likelihood_confidence_values.append(float(likelihood_confidence))

                    except (TypeError, ValueError) as e:
                        print(f"Warning: Could not process values in iteration {item['iteration']}: {e}")

                stats = {
                    'iteration': item['iteration'],
                    'impact_mean': np.mean(impact_values),
                    'impact_min': np.min(impact_values),
                    'impact_max': np.max(impact_values),
                    'impact_std': np.std(impact_values),
                    'impact_confidence_mean': np.mean(impact_confidence_values),
                    'impact_confidence_min': np.min(impact_confidence_values),
                    'impact_confidence_max': np.max(impact_confidence_values),
                    'impact_confidence_std': np.std(impact_confidence_values),
                    'likelihood_mean': np.mean(likelihood_values),
                    'likelihood_min': np.min(likelihood_values),
                    'likelihood_max': np.max(likelihood_values),
                    'likelihood_std': np.std(likelihood_values),
                    'likelihood_confidence_mean': np.mean(likelihood_confidence_values),
                    'likelihood_confidence_min': np.min(likelihood_confidence_values),
                    'likelihood_confidence_max': np.max(likelihood_confidence_values),
                    'likelihood_confidence_std': np.std(likelihood_confidence_values),
                    'num_risks': len(impact_values),
                    'similarity_score': item['scope_similarity_metrics']['similarity_score'],
                    'semantic_difference': item['scope_similarity_metrics']['semantic_difference'],
                    'jaccard_similarity': item['scope_similarity_metrics']['jaccard_similarity'],
                    'risk_similarity_matrix': item['risk_similarity_metrics']['similarity_matrix'],
                    'risk_avg_similarity': item['risk_similarity_metrics']['avg_similarity'],
                    'risk_max_similarity': item['risk_similarity_metrics']['max_similarity'],
                    'risk_min_similarity': item['risk_similarity_metrics']['min_similarity'],
                    'risk_avg_max_similarity': item['risk_similarity_metrics']['avg_max_similarity'],
                    'risk_avg_risk_drift': item['risk_similarity_metrics']['avg_risk_drift'],
                    'proportion_above_threshold': item['risk_similarity_metrics']['proportion_above_threshold'],
                    'max_similarities': item['risk_similarity_metrics']['max_similarities'],
                    'risk_drifts': item['risk_similarity_metrics']['risk_drifts']
                }

                impact_stats_per_iteration.append(stats)
            except Exception as e:
                print(f"Warning: Error processing iteration {item.get('iteration', 'unknown')}: {e}")
                raise Exception(f"Error processing iteration {item.get('iteration', 'unknown')}: {e}")
            
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(impact_stats_per_iteration)

        # Save statistics to CSV
        csv_filename = os.path.join(run_output_dir, f"risk_stats_iter.csv")
        df.to_csv(csv_filename, index=False)

        #font size must be atlesat 6 when resized in paper
        xlabel_fontsize = 12
        ylabel_fontsize = 12
        title_fontsize = 14
        tick_fontsize = 12
        legend_fontsize = 12
        if plot_now:
            # Plot 1: Impact Statistics
            plt.figure(figsize=(10, 6), dpi = 1200)
            plt.plot(df['iteration'], df['impact_mean'], 'g-o', label='Mean Impact')
            plt.fill_between(df['iteration'],
                            df['impact_mean'] - df['impact_std'],
                            df['impact_mean'] + df['impact_std'],
                            alpha=0.2, color='g', label='±1 Std Dev')
            plt.plot(df['iteration'], df['impact_max'], 'r--', label='Max Impact')
            plt.plot(df['iteration'], df['impact_min'], 'b--', label='Min Impact')
            plt.xlabel('Iteration', fontsize=xlabel_fontsize)
            plt.ylabel('Impact Level', fontsize=ylabel_fontsize)
            plt.title('Impact Statistics Across Iterations', fontsize=title_fontsize)
            plt.tick_params(axis='both', labelsize=tick_fontsize)
            plt.grid(True)
            plt.legend(fontsize=legend_fontsize)

            filename1 = os.path.join(run_output_dir, f"impact_stats_iter{last_iteration}.svg")
            plt.savefig(filename1, dpi=1200, bbox_inches='tight')
            plt.close()

            #Plot 1.1: Impact Confidence Statistics
            plt.figure(figsize=(10, 6), dpi = 1200)
            plt.plot(df['iteration'], df['impact_confidence_mean'], 'g-o', label='Mean Impact Confidence')
            plt.fill_between(df['iteration'],
                            df['impact_confidence_mean'] - df['impact_confidence_std'],
                            df['impact_confidence_mean'] + df['impact_confidence_std'],
                            alpha=0.2, color='g', label='±1 Std Dev')
            plt.plot(df['iteration'], df['impact_confidence_mean'], 'r--', label='Max Likelihood Confidence')
            plt.plot(df['iteration'], df['impact_confidence_mean'], 'b--', label='Min Likelihood Confidence')
            plt.xlabel('Iteration', fontsize=xlabel_fontsize)
            plt.ylabel('Impact Confidence', fontsize=ylabel_fontsize)
            plt.title('Impact Confidence Statistics Across Iterations', fontsize=title_fontsize)
            plt.tick_params(axis='both', labelsize=tick_fontsize)
            plt.grid(True)
            plt.legend(fontsize=legend_fontsize)

            filename1_1 = os.path.join(run_output_dir, f"impact_confidence_stats_iter{last_iteration}.svg")
            plt.savefig(filename1_1, dpi=1200, bbox_inches='tight')
            plt.close() 

            # Plot 2: Number of Risks per Iteration
            plt.figure(figsize=(10, 6), dpi = 1200)
            plt.plot(df['iteration'], df['num_risks'], 'b-o')
            plt.xlabel('Iteration', fontsize=xlabel_fontsize)
            plt.ylabel('Number of Risks', fontsize=ylabel_fontsize)
            plt.title('Number of Risks per Iteration', fontsize=title_fontsize)
            plt.tick_params(axis='both', labelsize=tick_fontsize)
            plt.grid(True)
            plt.legend(fontsize=legend_fontsize)

            filename2 = os.path.join(run_output_dir, f"num_risks_iter{last_iteration}.svg")
            plt.savefig(filename2, dpi=1200, bbox_inches='tight')
            plt.close()

            # Plot 3: Likelihood Statistics
            plt.figure(figsize=(10, 6), dpi = 1200)
            plt.plot(df['iteration'], df['likelihood_mean'], 'g-o', label='Mean Likelihood')
            plt.fill_between(df['iteration'],
                            df['likelihood_mean'] - df['likelihood_std'],
                            df['likelihood_mean'] + df['likelihood_std'],
                            alpha=0.2, color='g', label='±1 Std Dev')
            plt.plot(df['iteration'], df['likelihood_max'], 'r--', label='Max Likelihood')
            plt.plot(df['iteration'], df['likelihood_min'], 'b--', label='Min Likelihood')
            plt.xlabel('Iteration', fontsize=xlabel_fontsize)
            plt.ylabel('Likelihood Level', fontsize=ylabel_fontsize)
            plt.title('Likelihood Statistics Across Iterations', fontsize=title_fontsize)
            plt.tick_params(axis='both', labelsize=tick_fontsize)
            plt.grid(True)
            plt.legend(fontsize=legend_fontsize)

            filename3 = os.path.join(run_output_dir, f"likelihood_stats_iter{last_iteration}.svg")
            plt.savefig(filename3, dpi=1200, bbox_inches='tight')
            plt.close()

            #Plot 3.1: Likelihood Confidence Statistics
            plt.figure(figsize=(10, 6), dpi = 1200)
            plt.plot(df['iteration'], df['likelihood_confidence_mean'], 'g-o', label='Mean Likelihood Confidence')
            plt.fill_between(df['iteration'],
                            df['likelihood_confidence_mean'] - df['likelihood_confidence_std'],
                            df['likelihood_confidence_mean'] + df['likelihood_confidence_std'],
                            alpha=0.2, color='g', label='±1 Std Dev')
            plt.plot(df['iteration'], df['likelihood_confidence_max'], 'r--', label='Max Likelihood Confidence')
            plt.plot(df['iteration'], df['likelihood_confidence_min'], 'b--', label='Min Likelihood Confidence')
            plt.xlabel('Iteration', fontsize=xlabel_fontsize)
            plt.ylabel('Likelihood Confidence Level', fontsize=ylabel_fontsize)
            plt.title('Likelihood Confidence Statistics Across Iterations', fontsize=title_fontsize)
            plt.tick_params(axis='both', labelsize=tick_fontsize)
            plt.grid(True)
            plt.legend(fontsize=legend_fontsize)

            filename3_1 = os.path.join(run_output_dir, f"likelihood_confidence_stats_iter{last_iteration}.svg")
            plt.savefig(filename3_1, dpi=1200, bbox_inches='tight')
            plt.close()

            # Plot 4: Combined Impact and Likelihood Trends
            plt.figure(figsize=(12, 6), dpi = 1200)
            plt.plot(df['iteration'], df['impact_mean'], 'b-o', label='Mean Impact')
            plt.plot(df['iteration'], df['likelihood_mean'], 'r-o', label='Mean Likelihood')
            plt.fill_between(df['iteration'],
                            df['impact_mean'] - df['impact_std'],
                            df['impact_mean'] + df['impact_std'],
                            alpha=0.2, color='b', label='Impact ±1 Std Dev')
            plt.fill_between(df['iteration'],
                            df['likelihood_mean'] - df['likelihood_std'],
                            df['likelihood_mean'] + df['likelihood_std'],
                            alpha=0.2, color='r', label='Likelihood ±1 Std Dev')
            plt.xlabel('Iteration', fontsize=xlabel_fontsize)
            plt.ylabel('Level', fontsize=ylabel_fontsize)
            plt.title('Impact and Likelihood Trends Across Iterations', fontsize=title_fontsize)
            plt.tick_params(axis='both', labelsize=tick_fontsize)
            plt.grid(True)
            plt.legend(fontsize=legend_fontsize)
            filename4 = os.path.join(run_output_dir, f"combined_impact_likelihood_iter{last_iteration}.svg")
            plt.savefig(filename4, dpi=1200, bbox_inches='tight')
            plt.close()

            # Plot 5: Semantic Similarity Metrics
            plt.figure(figsize=(12, 6), dpi = 1200)
            plt.plot(df['iteration'], df['similarity_score'], 'b-o', label='Semantic Similarity Score')
            plt.xlabel('Iteration', fontsize=xlabel_fontsize)
            plt.ylabel('Cosine Similiarity Score', fontsize=ylabel_fontsize)
            plt.title('Semantic Similarity Metric Across Iterations', fontsize=title_fontsize)
            plt.tick_params(axis='both', labelsize=tick_fontsize)
            plt.grid(True)
            plt.legend(fontsize=legend_fontsize)

            filename5 = os.path.join(run_output_dir, f"semantic_similarity_metrics_iter{last_iteration}.svg")
            plt.savefig(filename5, dpi=1200, bbox_inches='tight')
            plt.close()

            # Plot 6: Semantic Difference Metrics
            plt.figure(figsize=(12, 6), dpi = 1200)
            plt.plot(df['iteration'], df['semantic_difference'], 'r-o', label='Semantic Difference')
            plt.xlabel('Iteration', fontsize=xlabel_fontsize)
            plt.ylabel('Cosine Distance', fontsize=ylabel_fontsize)
            plt.title('Semantic Difference Metric Across Iterations', fontsize=title_fontsize)
            plt.tick_params(axis='both', labelsize=tick_fontsize)
            plt.grid(True)
            plt.legend(fontsize=legend_fontsize)

            filename6 = os.path.join(run_output_dir, f"semantic_difference_metrics_iter{last_iteration}.svg")
            plt.savefig(filename6, dpi=1200, bbox_inches='tight')
            plt.close()

            #Plot 7: Jaccard Similarity Metrics
            plt.figure(figsize=(12, 6), dpi = 1200)
            plt.plot(df['iteration'], df['jaccard_similarity'], 'g-o', label='Jaccard Similarity')
            plt.xlabel('Iteration', fontsize=xlabel_fontsize)
            plt.ylabel('Jaccard Score', fontsize=ylabel_fontsize)
            plt.title('Jaccard Similairty Metrics Across Iterations', fontsize=title_fontsize)
            plt.tick_params(axis='both', labelsize=tick_fontsize)
            plt.grid(True)
            plt.legend(fontsize=legend_fontsize)

            filename7 = os.path.join(run_output_dir, f"jaccard_similarity_metrics_iter{last_iteration}.svg")
            plt.savefig(filename7, dpi=1200, bbox_inches='tight')
            plt.close()

            #Plot 8: Risk Similarity Metrics
            plt.figure(figsize=(12, 6), dpi = 1200)
            plt.plot(df['iteration'], df['risk_avg_similarity'], 'b-o', label='Avg Risk Similarity')
            plt.plot(df['iteration'], df['risk_max_similarity'], 'r--', label='Max Risk Similarity')
            plt.plot(df['iteration'], df['risk_min_similarity'], 'g--', label='Min Risk Similarity')
            plt.xlabel('Iteration', fontsize=xlabel_fontsize)
            plt.ylabel('Risks Cosine Similiarity Score', fontsize=ylabel_fontsize)
            plt.title('Risk Similarity Metrics Across Iterations', fontsize=title_fontsize)
            plt.tick_params(axis='both', labelsize=tick_fontsize)
            plt.grid(True)
            plt.legend(fontsize=legend_fontsize)

            filename8 = os.path.join(run_output_dir, f"risk_similarity_metrics_iter{last_iteration}.svg")
            plt.savefig(filename8, dpi=1200, bbox_inches='tight')
            plt.close()

            #Plot 9: Average Risk Drift
            plt.figure(figsize=(12, 6), dpi = 1200)
            plt.plot(df['iteration'], df['risk_avg_risk_drift'], 'b-o', label='Average Risk Drift')
            plt.xlabel('Iteration', fontsize=xlabel_fontsize)
            plt.ylabel('Average Risk Drift', fontsize=ylabel_fontsize)
            plt.title('Average Risk Drift Across Iterations', fontsize=title_fontsize)
            plt.tick_params(axis='both', labelsize=tick_fontsize)
            plt.grid(True)
            plt.legend(fontsize=legend_fontsize)

            filename9 = os.path.join(run_output_dir, f"avg_risk_drift_iter{last_iteration}.svg")
            plt.savefig(filename9, dpi=1200, bbox_inches='tight')
            plt.close()

            #Plot 10:  Proportion of Risks Above Threshold
            plt.figure(figsize=(12, 6), dpi = 1200)
            plt.plot(df['iteration'], df['proportion_above_threshold'], 'b-o', label='Proportion of Risks Above Threshold')
            plt.xlabel('Iteration', fontsize=xlabel_fontsize)
            plt.ylabel('Proportion', fontsize=ylabel_fontsize)
            plt.title('Proportion of Risks Above Threshold Across Iterations', fontsize=title_fontsize)
            plt.tick_params(axis='both', labelsize=tick_fontsize)
            plt.grid(True)
            plt.legend(fontsize=legend_fontsize)

            filename10 = os.path.join(run_output_dir, f"proportion_above_threshold_iter{last_iteration}.svg")
            plt.savefig(filename10, dpi=1200, bbox_inches='tight')
            plt.close()

            # Plot 11: Maximum Similarities
            plt.figure(figsize=(12, 6), dpi = 1200)
            for i, item in enumerate(self.iteration_history):
                plt.plot(item['risk_similarity_metrics']['max_similarities'], label=f'Risk {i+1}')
            plt.xlabel('Risk Index', fontsize=xlabel_fontsize)
            plt.ylabel('Similarity', fontsize=ylabel_fontsize)
            plt.title('Maximum Similarities Across Risks', fontsize=title_fontsize)
            plt.tick_params(axis='both', labelsize=tick_fontsize)
            plt.grid(True)
            plt.legend(fontsize=legend_fontsize)

            filename11 = os.path.join(run_output_dir, f"max_similarities_iter{last_iteration}.svg")
            plt.savefig(filename11, dpi=1200, bbox_inches='tight')
            plt.close()

            # Plot 12: Risk Drifts
            plt.figure(figsize=(12, 6), dpi = 1200)
            for i, item in enumerate(self.iteration_history):
                plt.plot(item['risk_similarity_metrics']['risk_drifts'], label=f'Risk {i+1}')
            plt.xlabel('Risk Index', fontsize=xlabel_fontsize)
            plt.ylabel('Drift', fontsize=ylabel_fontsize)
            plt.title('Risk Drifts Across Risks', fontsize=title_fontsize)
            plt.tick_params(axis='both', labelsize=tick_fontsize)
            plt.grid(True)
            plt.legend(fontsize=legend_fontsize)

            filename12 = os.path.join(run_output_dir, f"risk_drifts_iter{last_iteration}.svg")
            plt.savefig(filename12, dpi=1200, bbox_inches='tight')
            plt.close()

            #Plot I: Risk Similarity Matrix Heatmap (for each iteration)
            # Plot risk similarity matrices for each iteration
            for item in self.iteration_history:
                iteration_num = item['iteration']
                
                # Check if this iteration has risk similarity metrics
                if 'risk_similarity_metrics' in item and 'similarity_matrix' in item['risk_similarity_metrics']:
                    plt.figure(figsize=(10, 8), dpi=1200)
                    similarity_matrix = item['risk_similarity_metrics']['similarity_matrix']
                    risk_descriptions = item['risk_similarity_metrics'].get('risk_descriptions', [])
                    
                    # Create a heatmap of the similarity matrix
                    im = plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
                    plt.colorbar(im, label='Similarity Score')
                    
                    # Add labels and title
                    plt.title(f'Risk Similarity Matrix - Iteration {iteration_num}', fontsize=title_fontsize)
                    plt.xlabel('Risk Index', fontsize=xlabel_fontsize)
                    plt.ylabel('Risk Index', fontsize=ylabel_fontsize)
                    
                    # Add tick labels
                    n_risks = len(similarity_matrix)
                    plt.xticks(range(n_risks), range(n_risks), fontsize=tick_fontsize)
                    plt.yticks(range(n_risks), range(n_risks), fontsize=tick_fontsize)
                    
                    # Save the plot
                    matrix_filename = os.path.join(run_output_dir, f"risk_similarity_matrix_iter{iteration_num}.svg")
                    plt.savefig(matrix_filename, dpi=1200, bbox_inches='tight')
                    plt.close()
                    
                    # Create a text file with risk indices and descriptions for reference
                    desc_filename = os.path.join(run_output_dir, f"risk_descriptions_iter{iteration_num}.txt")
                    with open(desc_filename, 'w') as f:
                        for i, desc in enumerate(risk_descriptions):
                            # Truncate long descriptions for readability
                            truncated_desc = desc[:100] + "..." if len(desc) > 100 else desc
                            f.write(f"Risk {i}: {truncated_desc}\n")
                    
                    print(f"  - {os.path.basename(matrix_filename)}")
        
        # Print information about saved files
        print(f"\nResults saved in: {run_output_dir}")
        print(f"Files generated:")
        print(f"- Data: {os.path.basename(csv_filename)}")
        if plot_now:
            print(f"- Plots:")
            print(f"  - {os.path.basename(filename1)}")
            print(f"  - {os.path.basename(filename1_1)}")
            print(f"  - {os.path.basename(filename2)}")
            print(f"  - {os.path.basename(filename3)}")
            print(f"  - {os.path.basename(filename3_1)}")
            print(f"  - {os.path.basename(filename4)}")
            print(f"  - {os.path.basename(filename5)}")
            print(f"  - {os.path.basename(filename6)}")
            print(f"  - {os.path.basename(filename7)}")
            print(f"  - {os.path.basename(filename8)}")
            print(f"  - {os.path.basename(filename9)}")
            print(f"  - {os.path.basename(filename10)}")
            print(f"  - {os.path.basename(filename11)}")
            print(f"  - {os.path.basename(filename12)}")

        # Save agent settings to a file in the plot folder
        agent_settings_filename = os.path.join(run_output_dir, "agent_settings.json")
        with open(agent_settings_filename, "w") as f:
            json.dump(self.agent_settings, f, indent=4)
        print(f"  - {os.path.basename(agent_settings_filename)}")

def main(
        llm_client: str, 
        auto_answer: bool, 
        max_risks: int, 
        max_questions_per_risk: int, 
        max_iterations: int,
        initial_scope: str = "",
        answer_agent_settings: Dict = None,
        clarification_agent_settings: Dict = None,
        risk_identification_agent_settings: Dict = None,
        document_agent_settings: Dict = None
    ):

    # print("Enter your project scope (press Enter twice to finish):")
    #
    # scope_lines = []
    # while True:
    #     line = input()
    #     if not line and scope_lines:
    #         break
    #     if line:
    #         scope_lines.append(line)
    #
    # initial_scope = "\n".join(scope_lines)
    # print("Scope received, processing...")

    # Set the initial scope directly for testing purposes
    initial_scope = """
        The 100 TB data center is a state-of-the-art facility designed to provide 
        enterprise-grade storage and computing capabilities. The facility spans 10,000 
        square feet of climate-controlled space, featuring redundant power systems 
        with N+1 UPS configuration and diesel generators for backup power. The storage 
        infrastructure consists of high-performance storage arrays utilizing a combination 
        of NVMe SSDs for hot data and high-capacity enterprise HDDs for cold storage.

        The network architecture implements a leaf-spine topology with 100 Gbps 
        connectivity, ensuring low-latency data access and high availability. The 
        facility employs advanced cooling systems with hot/cold aisle containment 
        and precision air handling units maintaining an optimal PUE of 1.3. Security 
        measures include multi-factor authentication, 24/7 surveillance, and biometric 
        access controls.

        The storage system is configured with RAID 6 for data protection, featuring 
        automatic failover capabilities and real-time data replication across multiple 
        arrays. Regular backup procedures include both incremental and full backups, 
        with off-site replication for disaster recovery. The facility maintains 
        compliance with industry standards including SOC 2 Type II and ISO 27001, 
        ensuring data security and operational excellence.
        """

    answer_agent_settings = answer_agent_settings or {"temperature": 0, "min_tokens": 0, "max_tokens": 3000}
    clarification_agent_settings = clarification_agent_settings or {"temperature": 0, "min_tokens": 0, "max_tokens": 3000}
    risk_identification_agent_settings = risk_identification_agent_settings or {"temperature": 0, "min_tokens": 0, "max_tokens": 3000}
    document_agent_settings = document_agent_settings or {"temperature": 0, "min_tokens": 0, "max_tokens": 3000}

    print(f"Processing initial scope: {initial_scope}")

    try:
````````# if llm_client == "openai":
        #     model = "gpt-4.1-nano"
        #     coordinator = ScopingCoordinator(
        #         llm_client=OpenAIClient(),
        #         auto_answer=auto_answer,
        #         max_risks=max_risks,
        #         max_questions_per_risk=max_questions_per_risk,
        #         max_iterations=max_iterations
        #     )
        # elif llm_client == "huggingface":
        #     model = "tiiuae/falcon-7b-instruct"
        #     coordinator = ScopingCoordinator(
        #         llm_client=HuggingFaceClient(model=model),
        #         auto_answer=auto_answer,
        #         max_risks=max_risks,
        #         max_questions_per_risk=max_questions_per_risk,
        #         max_iterations=max_iterations
        #     )

        if llm_client == "openai":
            answer_llm = OpenAIClient(model="gpt-4.1-nano", **answer_agent_settings)
            clarification_llm = OpenAIClient(model="gpt-4.1-nano", **clarification_agent_settings)
            risk_identification_llm = OpenAIClient(model="gpt-4.1-nano", **risk_identification_agent_settings)
            document_llm = OpenAIClient(model="gpt-4.1-nano", **document_agent_settings)
        elif llm_client == "huggingface":
            answer_llm = HuggingFaceClient(**answer_agent_settings)
            clarification_llm = HuggingFaceClient(**clarification_agent_settings)
            risk_identification_llm = HuggingFaceClient(**risk_identification_agent_settings)
            document_llm = HuggingFaceClient(**document_agent_settings)
        else:
            raise ValueError(f"Unsupported LLM client type: {llm_client}")

        coordinator = ScopingCoordinator(
            answer_llm=answer_llm,
            clarification_llm=clarification_llm,
            risk_identification_llm=risk_identification_llm,
            document_llm=document_llm,

            auto_answer=auto_answer,
            max_risks=max_risks,
            max_questions_per_risk=max_questions_per_risk,
            max_iterations=max_iterations,

            agent_settings = {
                "answer_agent": answer_agent_settings,
                "clarification_agent": clarification_agent_settings,
                "risk_identification_agent": risk_identification_agent_settings,
                "document_agent": document_agent_settings
            }
    )

        # print("LLM model:", model)
        result = coordinator.process_request(initial_scope)

        if result["status"] == "completed":
            # print("\n=== Final Results ===")

            # Print Top Risk Items
            # print("\nTop Risk Items Analyzed:")
            # print("=" * 80)
            # for i, risk in enumerate(result["risks"][:max_risks], 1):
            #     print(f"\n{i}. Risk Category: {risk['category']}")
            #     print(f"   Description: {risk['risk_description']}")
            #     print(f"   Risk Score: {risk['risk_score']}")
            #     print(f"   Impact: {risk['impact']['level']} ({risk['impact']['value']})")
            #     print(f"   Likelihood: {risk['likelihood']['level']} ({risk['likelihood']['value']})")
            #     print("-" * 80)

            # Print Questions and Answers
            # if "clarifications" in result:
            #     print("\nClarification Questions and Answers:")
            #     print("=" * 80)
            #     for i, clarification in enumerate(result["clarifications"], 1):
            #         print(f"\nRisk Category: {clarification['risk_category']}")
            #         print(f"Risk Description: {clarification['risk_description']}")
            #         print("\nQuestions and Answers:")
            #         for j, (question, answer) in enumerate(zip(
            #                 clarification.get('questions', []),
            #                 clarification.get('answers', [])), 1):
            #             print(f"\n   Q{j}: {question}")
            #             print(f"   A{j}: {answer}")
            #         print("-" * 80)

            print(f"\n{'#' * 30}")
            print("Final Scope:")
            print(f"{'#' * 30}\n")
            print(result["final_scope"])

            # print("\nFinal Document:")
            # print(result["final_document"])
        else:
            print("\nError:", result["error"])

    except Exception as e:
        print(f"\nError during processing: {str(e)}")

if __name__ == "__main__":
    # predict_risk("this is an example risk")
    
    main(llm_client="openai", auto_answer=True, max_risks=1, max_questions_per_risk=1, max_iterations=2)

