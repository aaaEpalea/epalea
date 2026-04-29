"""
Schema Definition Module
Defines predicates, variables, and their relationships for SPN reasoning.
"""
import json
from typing import Dict, List, Set
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Variable:
    name: str
    domain: List[str]
    var_type: str = "categorical"
    def __post_init__(self):
        if self.var_type == "categorical" and not self.domain:
            raise ValueError(f"Categorical variable {self.name} must have non-empty domain")
    def __hash__(self): return hash(self.name)
    def __eq__(self, other): return isinstance(other, Variable) and self.name == other.name


@dataclass
class Predicate:
    name: str
    variables: List[str]
    domain: List[str]
    description: str = ""
    def __hash__(self): return hash(self.name)
    def __eq__(self, other): return isinstance(other, Predicate) and self.name == other.name


class Schema:
    def __init__(self):
        self.variables: Dict[str, Variable] = {}
        self.predicates: Dict[str, Predicate] = {}
        self._predicate_to_vars: Dict[str, Set[str]] = {}

    def add_variable(self, name: str, domain: List[str], var_type: str = "categorical") -> Variable:
        if name in self.variables:
            raise ValueError(f"Variable {name} already exists")
        var = Variable(name=name, domain=domain, var_type=var_type)
        self.variables[name] = var
        return var

    def add_predicate(self, name: str, variables: List[str], domain: List[str], description: str = "") -> Predicate:
        if name in self.predicates:
            raise ValueError(f"Predicate {name} already exists")
        for var_name in variables:
            if var_name not in self.variables:
                raise ValueError(f"Variable {var_name} not found in schema")
        pred = Predicate(name=name, variables=variables, domain=domain, description=description)
        self.predicates[name] = pred
        self._predicate_to_vars[name] = set(variables)
        return pred

    def get_variables_for_predicate(self, predicate_name: str) -> List[str]:
        if predicate_name not in self.predicates:
            raise ValueError(f"Predicate {predicate_name} not found")
        return self.predicates[predicate_name].variables.copy()

    def get_predicate_domain(self, predicate_name: str) -> List[str]:
        if predicate_name not in self.predicates:
            raise ValueError(f"Predicate {predicate_name} not found")
        return self.predicates[predicate_name].domain.copy()

    def get_variable_domain(self, variable_name: str) -> List[str]:
        if variable_name not in self.variables:
            raise ValueError(f"Variable {variable_name} not found")
        return self.variables[variable_name].domain.copy()

    def covers_predicate(self, predicate_name: str) -> bool:
        return predicate_name in self.predicates

    def get_related_predicates(self, variable_name: str) -> List[str]:
        if variable_name not in self.variables:
            raise ValueError(f"Variable {variable_name} not found")
        return [p for p, vs in self._predicate_to_vars.items() if variable_name in vs]

    def save(self, path: str) -> None:
        """Save schema to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "variables": {
                name: {"domain": v.domain, "var_type": v.var_type}
                for name, v in self.variables.items()
            },
            "predicates": {
                name: {"variables": p.variables, "domain": p.domain, "description": p.description}
                for name, p in self.predicates.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Schema":
        """Load schema from JSON file."""
        with open(path) as f:
            data = json.load(f)
        schema = cls()
        for name, v in data.get("variables", {}).items():
            schema.add_variable(name, v["domain"], v.get("var_type", "categorical"))
        for name, p in data.get("predicates", {}).items():
            schema.add_predicate(name, p["variables"], p["domain"], p.get("description", ""))
        return schema

    def __repr__(self) -> str:
        return f"Schema(variables={len(self.variables)}, predicates={len(self.predicates)})"
