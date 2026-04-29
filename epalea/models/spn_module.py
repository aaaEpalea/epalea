"""
Lightweight SPN Module
Implements Sum-Product Networks for tractable probabilistic reasoning.
Built from scratch in PyTorch for stability and transparency.

Section 3.2: SPNs provide tractable exact inference through recursive computation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Any
from abc import ABC, abstractmethod
import numpy as np
from epalea.models.schema import Schema


class SPNNode(ABC, nn.Module):
    """
    Base class for all SPN nodes.
    
    An SPN is a rooted DAG where:
    - Leaves represent probability distributions over variables
    - Internal nodes are sums or products
    - The network computes a valid probability distribution
    """
    
    def __init__(self, scope: Set[str]):
        """
        Initialize SPN node.
        
        Args:
            scope: Set of variable names this node depends on
        """
        super().__init__()
        self.scope = scope
    
    @abstractmethod
    def forward(self, evidence: Dict[str, Any]) -> torch.Tensor:
        """
        Compute (unnormalized) log probability given evidence.
        
        Args:
            evidence: Dictionary mapping variable names to values or None
            
        Returns:
            Log probability tensor of shape (batch_size,)
        """
        pass
    
    def eval_log_prob(self, evidence: Dict[str, Any]) -> torch.Tensor:
        """Evaluate log probability (alias for forward)."""
        return self.forward(evidence)


class LeafNode(SPNNode):
    """
    Leaf node representing a categorical distribution over a single variable.
    
    For variable V with domain {v1, v2, ..., vk}, stores probabilities
    P(V=vi) for each value.
    """
    
    def __init__(self, variable_name: str, domain: List[str]):
        """
        Initialize leaf node.
        
        Args:
            variable_name: Name of the variable
            domain: List of possible values for the variable
        """
        super().__init__(scope={variable_name})
        self.variable_name = variable_name
        self.domain = domain
        self.value_to_idx = {val: idx for idx, val in enumerate(domain)}
        
        # Initialize with uniform distribution
        # Use log probabilities for numerical stability
        n = len(domain)
        self.log_probs = nn.Parameter(
            torch.log(torch.ones(n) / n),
            requires_grad=False  # Typically not learned in our setting
        )
    
    def set_distribution(self, distribution: Dict[str, float]):
        """
        Set the probability distribution.
        
        Args:
            distribution: Dict mapping values to probabilities
        """
        probs = torch.zeros(len(self.domain))
        for value, prob in distribution.items():
            if value in self.value_to_idx:
                idx = self.value_to_idx[value]
                probs[idx] = prob
        
        # Normalize
        probs = probs / (probs.sum() + 1e-10)
        
        # Set log probabilities
        self.log_probs.data = torch.log(probs + 1e-10)
    
    def forward(self, evidence: Dict[str, Any]) -> torch.Tensor:
        """
        Evaluate log probability given evidence.
        
        If variable is observed, return log P(V=observed_value).
        If variable is not observed, return log 1 (marginalize).
        
        Args:
            evidence: Dictionary of observed values
            
        Returns:
            Log probability (scalar)
        """
        if self.variable_name in evidence:
            value = evidence[self.variable_name]
            if value in self.value_to_idx:
                idx = self.value_to_idx[value]
                return self.log_probs[idx].unsqueeze(0)
            else:
                # Unknown value, return very low probability
                return torch.tensor([-1e10])
        else:
            # Variable not observed, marginalize (return 0 in log space)
            return torch.tensor([0.0])


class ProductNode(SPNNode):
    """
    Product node representing factorization over disjoint scopes.
    
    For children with scopes S1, S2, ..., Sk (disjoint):
        P(S1 ∪ S2 ∪ ... ∪ Sk) = P(S1) × P(S2) × ... × P(Sk)
    
    In log space: log P = Σ log P(Si)
    """
    
    def __init__(self, children: List[SPNNode]):
        """
        Initialize product node.
        
        Args:
            children: List of child nodes
        """
        # Union of children's scopes
        scope = set()
        for child in children:
            scope.update(child.scope)
        
        super().__init__(scope=scope)
        self._children = nn.ModuleList(children)
    
    def forward(self, evidence: Dict[str, Any]) -> torch.Tensor:
        """
        Compute product: sum of log probabilities.
        
        Args:
            evidence: Dictionary of observed values
            
        Returns:
            Log probability (sum of children's log probs)
        """
        log_prob = torch.tensor([0.0])
        for child in self._children:
            log_prob = log_prob + child(evidence)
        return log_prob


class SumNode(SPNNode):
    """
    Sum node representing mixture over alternatives.
    
    For children with same scope S and weights w1, w2, ..., wk:
        P(S) = w1·P1(S) + w2·P2(S) + ... + wk·Pk(S)
    
    In log space: log P = log(Σ wi × exp(log Pi))
                        = logsumexp(log wi + log Pi)
    """
    
    def __init__(self, children: List[SPNNode], weights: Optional[List[float]] = None):
        """
        Initialize sum node.
        
        Args:
            children: List of child nodes (must have same scope)
            weights: Mixture weights (default: uniform)
        """
        # All children must have same scope
        if children:
            scope = children[0].scope
            for child in children[1:]:
                assert child.scope == scope, "Sum node children must have same scope"
        else:
            scope = set()
        
        super().__init__(scope=scope)
        self._children = nn.ModuleList(children)
        
        # Initialize weights
        if weights is None:
            weights = [1.0 / len(children)] * len(children)
        
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        weights_tensor = weights_tensor / weights_tensor.sum()  # Normalize
        
        self.log_weights = nn.Parameter(
            torch.log(weights_tensor),
            requires_grad=False
        )
    
    def forward(self, evidence: Dict[str, Any]) -> torch.Tensor:
        """
        Compute mixture: logsumexp of weighted children.
        
        Args:
            evidence: Dictionary of observed values
            
        Returns:
            Log probability (logsumexp of weighted children)
        """
        if not self._children:
            return torch.tensor([0.0])
        
        # Collect log probs from children
        log_probs = []
        for i, child in enumerate(self._children):
            child_log_prob = child(evidence)
            weighted_log_prob = self.log_weights[i] + child_log_prob
            log_probs.append(weighted_log_prob)
        
        # Stack and logsumexp
        log_probs_tensor = torch.stack(log_probs)
        return torch.logsumexp(log_probs_tensor, dim=0)

class LikelihoodNode(SPNNode):
    """
    Likelihood node for attaching soft factors from evidence.
    
    This represents the soft likelihood Φ_e(y) from Section 4.1.
    Can be attached dynamically at query time.
    """
    
    def __init__(
        self,
        variable_name: str,
        domain: List[str],
        potential: Optional[Dict[str, float]] = None,
        weight: float = 1.0
    ):
        """
        Initialize likelihood node.
        
        Args:
            variable_name: Variable this likelihood is over
            domain: Possible values
            potential: Initial potential (default: uniform)
            weight: Credibility weight
        """
        super().__init__(scope={variable_name})
        self.variable_name = variable_name
        self.domain = domain
        self.value_to_idx = {val: idx for idx, val in enumerate(domain)}
        self.weight = weight
        
        # Initialize potential
        if potential is None:
            potential = {val: 1.0 / len(domain) for val in domain}
        
        self.set_potential(potential, weight)
    
    def set_potential(self, potential: Dict[str, float], weight: float = 1.0):
        """
        Set the likelihood potential.
        
        Args:
            potential: Distribution over variable values
            weight: Credibility weight (Section 4.2)
        """
        self.weight = weight
        
        # Convert to tensor
        probs = torch.zeros(len(self.domain))
        for value, prob in potential.items():
            if value in self.value_to_idx:
                idx = self.value_to_idx[value]
                probs[idx] = prob
        
        # Normalize
        probs = probs / (probs.sum() + 1e-10)
        
        # Apply weight: p^w (Section 5.6)
        weighted_probs = probs ** weight
        weighted_probs = weighted_probs / (weighted_probs.sum() + 1e-10)
        
        # Store as log probabilities
        self.log_probs = torch.log(weighted_probs + 1e-10)
    
    def forward(self, evidence: Dict[str, Any]) -> torch.Tensor:
        """
        Evaluate likelihood.
        
        Args:
            evidence: Dictionary of observed values
            
        Returns:
            Log likelihood
        """
        if self.variable_name in evidence:
            value = evidence[self.variable_name]
            if value in self.value_to_idx:
                idx = self.value_to_idx[value]
                return self.log_probs[idx].unsqueeze(0)
            else:
                return torch.tensor([-1e10])
        else:
            # Not observed, return 0 (no constraint)
            return torch.tensor([0.0])


class SPN:
    """
    Sum-Product Network for tractable probabilistic inference.
    
    Implements Algorithm 3: SPNModule.Query from the paper.
    """
    
    def __init__(self, schema: Schema):
        """
        Initialize SPN.
        
        Args:
            schema: Schema defining variables and predicates
        """
        self.schema = schema
        self.root: Optional[SPNNode] = None
        self.likelihood_nodes: Dict[str, List[LikelihoodNode]] = {}
    
    def build_simple_structure(self, predicate: str):
        """
        Build a simple SPN structure for a predicate.
        
        Creates a flat structure:
        - Root: SumNode with two branches (low/high compliance scenarios)
        - Each branch: ProductNode over predicate variables
        - Leaves: Categorical distributions over variable domains
        
        Args:
            predicate: Predicate name to build structure for
        """
        if not self.schema.covers_predicate(predicate):
            raise ValueError(f"Predicate {predicate} not in schema")
        
        variables = self.schema.get_variables_for_predicate(predicate)
        
        # Create leaf nodes for each variable
        leaves = []
        for var_name in variables:
            domain = self.schema.get_variable_domain(var_name)
            leaf = LeafNode(var_name, domain)
            leaves.append(leaf)
        
        # Create product node over leaves (factorization)
        if len(leaves) > 1:
            product = ProductNode(leaves)
            # Create sum node with single child (simple structure)
            self.root = SumNode([product], weights=[1.0])
        elif len(leaves) == 1:
            self.root = SumNode([leaves[0]], weights=[1.0])
        else:
            # No variables, just a dummy root
            self.root = SumNode([], weights=[])
    
    def attach_soft_factors(self, factors: List[Any]):
        """
        Attach soft likelihood factors from evidence.
        Implements lines 9-14 of Algorithm 3.
        
        Args:
            factors: List of SoftFactor objects from FactorConverter
        """
        from epalea.core.factor_converter import SoftFactor
        
        for factor in factors:
            if not isinstance(factor, SoftFactor):
                continue
            
            # Get the variable this factor is over
            # For simplicity, assume factor is over a single predicate variable
            variables = factor.variables
            if not variables:
                continue
            
            # Create likelihood node
            # Get domain from schema
            var_name = variables[0]  # Simplified: use first variable
            if var_name not in self.schema.variables:
                continue
            
            domain = self.schema.get_variable_domain(var_name)
            
            likelihood = LikelihoodNode(
                variable_name=var_name,
                domain=domain,
                potential=factor.potential,
                weight=factor.weight
            )
            
            # Store for later use
            if var_name not in self.likelihood_nodes:
                self.likelihood_nodes[var_name] = []
            self.likelihood_nodes[var_name].append(likelihood)
    
    def query(
        self,
        query_variable: str,
        evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Perform marginal inference over query variable.
        Implements Algorithm 3: SPNModule.Query
        
        Args:
            query_variable: Variable to compute marginal for
            evidence: Dictionary of observed variable values
            
        Returns:
            Probability distribution over query variable values
        """
        if self.root is None:
            raise ValueError("SPN structure not built. Call build_simple_structure first.")
        
        if evidence is None:
            evidence = {}
        
        # Get domain for query variable
        if query_variable not in self.schema.variables:
            raise ValueError(f"Query variable {query_variable} not in schema")
        
        domain = self.schema.get_variable_domain(query_variable)
        
        # Compute marginal by conditioning on each value
        marginal = {}
        log_probs = []
        
        for value in domain:
            # Set evidence for this value
            evidence_with_query = evidence.copy()
            evidence_with_query[query_variable] = value
            
            # Evaluate SPN
            log_prob = self.root(evidence_with_query)
            
            # Add likelihoods from soft factors
            if query_variable in self.likelihood_nodes:
                for likelihood in self.likelihood_nodes[query_variable]:
                    log_prob = log_prob + likelihood(evidence_with_query)
            
            log_probs.append(log_prob)
        
        # Convert to probabilities and normalize
        log_probs_tensor = torch.stack(log_probs)
        probs = torch.exp(log_probs_tensor)
        probs = probs / (probs.sum() + 1e-10)
        
        # Create result dictionary
        for value, prob in zip(domain, probs):
            marginal[value] = float(prob)
        
        return marginal
    
    def set_evidence(self, variable: str, value: Any):
        """
        Set hard evidence for a variable.
        
        Args:
            variable: Variable name
            value: Observed value
        """
        # This is handled in query() via evidence dict
        pass
    
    def covers_predicate(self, predicate: str) -> bool:
        """
        Check if SPN covers a predicate.
        
        Args:
            predicate: Predicate name
            
        Returns:
            True if predicate is covered
        """
        return self.schema.covers_predicate(predicate)
    
    def reset_likelihoods(self):
        """Clear all attached likelihood nodes."""
        self.likelihood_nodes.clear()


def build_spn_for_schema(schema: Schema, predicate: str) -> SPN:
    """
    Build a simple SPN structure for a predicate.
    
    Args:
        schema: Schema defining predicates and variables
        predicate: Predicate to build SPN for
        
    Returns:
        Initialized SPN
    """
    spn = SPN(schema)
    spn.build_simple_structure(predicate)
    return spn
