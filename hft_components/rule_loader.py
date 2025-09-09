#!/usr/bin/env python3
"""
Rule Loader and Validator for HFT Neurosymbolic AI System
Loads, validates, and manages trading rule packs in YAML format
"""

import yaml
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import jsonschema
from jsonschema import validate, ValidationError
import json

logger = logging.getLogger(__name__)

class RuleLoader:
    """Loads and validates neurosymbolic trading rule packs"""
    
    def __init__(self, rules_dir: str = "config"):
        self.rules_dir = Path(rules_dir)
        self.rules_cache = {}
        self.schema = None
        self._load_schema()
        
    def _load_schema(self):
        """Load the rule pack schema"""
        try:
            schema_path = self.rules_dir / "rules_schema.yaml"
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    self.schema = yaml.safe_load(f)
                logger.info("Rule schema loaded successfully")
            else:
                logger.warning("Rule schema not found, validation will be limited")
                self.schema = {}
        except Exception as e:
            logger.error(f"Failed to load rule schema: {e}")
            self.schema = {}
    
    def load_rule_pack(self, filename: str) -> Dict[str, Any]:
        """Load a rule pack from YAML file"""
        try:
            file_path = self.rules_dir / filename
            
            if not file_path.exists():
                raise FileNotFoundError(f"Rule pack file not found: {file_path}")
            
            with open(file_path, 'r') as f:
                rule_pack = yaml.safe_load(f)
            
            # Validate the rule pack
            if self.schema:
                validation_result = self.validate_rule_pack(rule_pack)
                if not validation_result['valid']:
                    logger.error(f"Rule pack validation failed: {validation_result['errors']}")
                    raise ValueError(f"Invalid rule pack: {validation_result['errors']}")
            
            # Cache the rule pack
            rule_pack_name = rule_pack.get('metadata', {}).get('name', filename)
            self.rules_cache[rule_pack_name] = rule_pack
            
            logger.info(f"Rule pack loaded successfully: {rule_pack_name}")
            return rule_pack
            
        except Exception as e:
            logger.error(f"Failed to load rule pack {filename}: {e}")
            raise
    
    def validate_rule_pack(self, rule_pack: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a rule pack against the schema"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Basic structure validation
            if not self._validate_required_fields(rule_pack, validation_result):
                validation_result['valid'] = False
            
            # Data type validation
            if not self._validate_data_types(rule_pack, validation_result):
                validation_result['valid'] = False
            
            # Value range validation
            if not self._validate_value_ranges(rule_pack, validation_result):
                validation_result['valid'] = False
            
            # Dependency validation
            if not self._validate_dependencies(rule_pack, validation_result):
                validation_result['valid'] = False
            
            # Logical consistency validation
            if not self._validate_logical_consistency(rule_pack, validation_result):
                validation_result['valid'] = False
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {e}")
        
        return validation_result
    
    def _validate_required_fields(self, rule_pack: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Validate that all required fields are present"""
        if not self.schema or 'validation' not in self.schema:
            return True
        
        required_fields = self.schema['validation'].get('required_fields', [])
        valid = True
        
        for field_path in required_fields:
            if not self._field_exists(rule_pack, field_path):
                result['errors'].append(f"Missing required field: {field_path}")
                valid = False
        
        return valid
    
    def _validate_data_types(self, rule_pack: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Validate data types of fields"""
        if not self.schema or 'validation' not in self.schema:
            return True
        
        data_types = self.schema['validation'].get('data_types', {})
        valid = True
        
        for field_path, expected_type in data_types.items():
            try:
                value = self._get_field_value(rule_pack, field_path)
                if value is not None:
                    if not self._check_type(value, expected_type):
                        result['errors'].append(
                            f"Invalid type for {field_path}: expected {expected_type}, got {type(value).__name__}"
                        )
                        valid = False
            except (KeyError, TypeError, IndexError):
                # Field doesn't exist, skip validation
                continue
        
        return valid
    
    def _validate_value_ranges(self, rule_pack: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Validate value ranges for numeric fields"""
        if not self.schema or 'validation' not in self.schema:
            return True
        
        value_ranges = self.schema['validation'].get('value_ranges', {})
        valid = True
        
        for field_path, (min_val, max_val) in value_ranges.items():
            try:
                value = self._get_field_value(rule_pack, field_path)
                if value is not None and isinstance(value, (int, float)):
                    if not (min_val <= value <= max_val):
                        result['errors'].append(
                            f"Value out of range for {field_path}: {value} not in [{min_val}, {max_val}]"
                        )
                        valid = False
            except (KeyError, TypeError, IndexError):
                # Field doesn't exist, skip validation
                continue
        
        return valid
    
    def _validate_dependencies(self, rule_pack: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Validate dependencies between fields"""
        if not self.schema or 'validation' not in self.schema:
            return True
        
        dependencies = self.schema['validation'].get('dependencies', [])
        valid = True
        
        for dependency in dependencies:
            if not self._check_dependency(rule_pack, dependency, result):
                valid = False
        
        return valid
    
    def _validate_logical_consistency(self, rule_pack: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Validate logical consistency of rules"""
        valid = True
        
        # Check for conflicting market regime conditions
        if 'market_regimes' in rule_pack:
            valid &= self._check_market_regime_consistency(rule_pack['market_regimes'], result)
        
        # Check for overlapping technical signal conditions
        if 'technical_signals' in rule_pack:
            valid &= self._check_technical_signal_consistency(rule_pack['technical_signals'], result)
        
        return valid
    
    def _check_market_regime_consistency(self, regimes: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Check for logical consistency in market regime definitions"""
        valid = True
        
        # Check that no two regimes have identical conditions
        regime_conditions = []
        for regime_name, regime_data in regimes.items():
            if 'conditions' in regime_data:
                conditions = tuple(sorted(str(c) for c in regime_data['conditions']))
                if conditions in regime_conditions:
                    result['warnings'].append(
                        f"Market regime {regime_name} has identical conditions to another regime"
                    )
                regime_conditions.append(conditions)
        
        return valid
    
    def _check_technical_signal_consistency(self, signals: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Check for logical consistency in technical signal definitions"""
        valid = True
        
        # Check for conflicting signal conditions within the same signal type
        for signal_name, signal_data in signals.items():
            if 'rules' in signal_data:
                rules = signal_data['rules']
                for rule_name, rule_data in rules.items():
                    if 'condition' in rule_data:
                        # Basic check for obvious conflicts (could be expanded)
                        condition = rule_data['condition']
                        if 'AND' in condition and 'OR' in condition:
                            result['warnings'].append(
                                f"Complex condition in {signal_name}.{rule_name}: {condition}"
                            )
        
        return valid
    
    def _field_exists(self, obj: Any, field_path: str) -> bool:
        """Check if a field exists at the given path"""
        try:
            self._get_field_value(obj, field_path)
            return True
        except (KeyError, TypeError, IndexError):
            return False
    
    def _get_field_value(self, obj: Any, field_path: str) -> Any:
        """Get the value of a field at the given path"""
        keys = field_path.split('.')
        current = obj
        
        for key in keys:
            if isinstance(current, dict):
                current = current[key]
            elif isinstance(current, list):
                current = current[int(key)]
            else:
                raise KeyError(f"Cannot access {key} in {type(current)}")
        
        return current
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if a value matches the expected type"""
        type_mapping = {
            'float': (int, float),
            'int': int,
            'string': str,
            'bool': bool,
            'list': list,
            'dict': dict
        }
        
        if expected_type in type_mapping:
            return isinstance(value, type_mapping[expected_type])
        
        return True  # Unknown type, assume valid
    
    def _check_dependency(self, rule_pack: Dict[str, Any], dependency: str, result: Dict[str, Any]) -> bool:
        """Check if a dependency is satisfied"""
        # This is a simplified dependency checker
        # In a full implementation, you'd want more sophisticated logic
        if "must be" in dependency:
            # Extract the condition from the dependency string
            # This is a basic implementation - could be enhanced with proper parsing
            result['warnings'].append(f"Dependency check not fully implemented: {dependency}")
            return True
        
        return True
    
    def get_rule_pack(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a cached rule pack by name"""
        return self.rules_cache.get(name)
    
    def list_rule_packs(self) -> List[str]:
        """List all available rule pack names"""
        return list(self.rules_cache.keys())
    
    def reload_rule_packs(self):
        """Reload all rule packs from disk"""
        self.rules_cache.clear()
        
        if self.rules_dir.exists():
            for yaml_file in self.rules_dir.glob("*.yaml"):
                if yaml_file.name != "rules_schema.yaml":
                    try:
                        self.load_rule_pack(yaml_file.name)
                    except Exception as e:
                        logger.warning(f"Failed to reload rule pack {yaml_file.name}: {e}")
    
    def get_market_regime_rules(self, rule_pack_name: str) -> Dict[str, Any]:
        """Get market regime rules from a specific rule pack"""
        rule_pack = self.get_rule_pack(rule_pack_name)
        if rule_pack and 'market_regimes' in rule_pack:
            return rule_pack['market_regimes']
        return {}
    
    def get_technical_signal_rules(self, rule_pack_name: str) -> Dict[str, Any]:
        """Get technical signal rules from a specific rule pack"""
        rule_pack = self.get_rule_pack(rule_pack_name)
        if rule_pack and 'technical_signals' in rule_pack:
            return rule_pack['technical_signals']
        return {}
    
    def get_risk_assessment_rules(self, rule_pack_name: str) -> Dict[str, Any]:
        """Get risk assessment rules from a specific rule pack"""
        rule_pack = self.get_rule_pack(rule_pack_name)
        if rule_pack and 'risk_assessment' in rule_pack:
            return rule_pack['risk_assessment']
        return {}
    
    def get_compliance_rules(self, rule_pack_name: str) -> Dict[str, Any]:
        """Get compliance rules from a specific rule pack"""
        rule_pack = self.get_rule_pack(rule_pack_name)
        if rule_pack and 'compliance' in rule_pack:
            return rule_pack['compliance']
        return {}
    
    def get_execution_rules(self, rule_pack_name: str) -> Dict[str, Any]:
        """Get execution rules from a specific rule pack"""
        rule_pack = self.get_rule_pack(rule_pack_name)
        if rule_pack and 'execution' in rule_pack:
            return rule_pack['execution']
        return {}
    
    def export_rule_pack(self, rule_pack_name: str, output_path: str) -> bool:
        """Export a rule pack to a file"""
        try:
            rule_pack = self.get_rule_pack(rule_pack_name)
            if not rule_pack:
                return False
            
            with open(output_path, 'w') as f:
                yaml.dump(rule_pack, f, default_flow_style=False, indent=2)
            
            logger.info(f"Rule pack {rule_pack_name} exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export rule pack {rule_pack_name}: {e}")
            return False
    
    def create_rule_pack_template(self, output_path: str) -> bool:
        """Create a template rule pack file"""
        try:
            template = {
                'metadata': {
                    'name': 'template_rule_pack',
                    'version': '1.0.0',
                    'description': 'Template for neurosymbolic trading rules',
                    'author': 'Your Name',
                    'created': 'YYYY-MM-DD',
                    'last_updated': 'YYYY-MM-DD',
                    'tags': ['trading', 'neurosymbolic', 'hft']
                },
                'market_regimes': {
                    'example_regime': {
                        'name': 'Example Market Regime',
                        'description': 'Description of the regime',
                        'conditions': [
                            {'condition_name': 'condition_value'}
                        ],
                        'confidence_threshold': 0.7,
                        'actions': [
                            {
                                'action': 'buy',
                                'confidence': 0.8,
                                'risk_level': 'moderate'
                            }
                        ]
                    }
                },
                'technical_signals': {},
                'risk_assessment': {},
                'compliance': {},
                'execution': {}
            }
            
            with open(output_path, 'w') as f:
                yaml.dump(template, f, default_flow_style=False, indent=2)
            
            logger.info(f"Rule pack template created at {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create rule pack template: {e}")
            return False

def main():
    """Test the rule loader"""
    loader = RuleLoader()
    
    # Create a template
    loader.create_rule_pack_template("config/template_rules.yaml")
    
    # Load the default rules
    try:
        rules = loader.load_rule_pack("rules_schema.yaml")
        print(f"Loaded rule pack: {rules['metadata']['name']}")
        print(f"Available rule packs: {loader.list_rule_packs()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
