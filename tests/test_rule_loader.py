#!/usr/bin/env python3
"""
Tests for Rule Loader and Validator
"""

import unittest
import tempfile
import os
import yaml
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hft_components.rule_loader import RuleLoader

class TestRuleLoader(unittest.TestCase):
    """Test cases for RuleLoader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.loader = RuleLoader(self.test_dir)
        
        # Create a test rule pack
        self.test_rules = {
            'metadata': {
                'name': 'test_rules',
                'version': '1.0.0',
                'description': 'Test rule pack',
                'author': 'Test Author',
                'created': '2025-08-29',
                'last_updated': '2025-08-29',
                'tags': ['test', 'trading']
            },
            'market_regimes': {
                'test_regime': {
                    'name': 'Test Market Regime',
                    'description': 'A test regime',
                    'conditions': [
                        {'test_condition': 'test_value'}
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
        
        # Create a test schema
        self.test_schema = {
            'validation': {
                'required_fields': [
                    'metadata.name',
                    'metadata.version',
                    'market_regimes'
                ],
                'data_types': {
                    'metadata.version': 'string',
                    'market_regimes.test_regime.confidence_threshold': 'float'
                },
                'value_ranges': {
                    'market_regimes.test_regime.confidence_threshold': [0.0, 1.0]
                },
                'dependencies': []
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_create_rule_pack_template(self):
        """Test creating a rule pack template"""
        template_path = os.path.join(self.test_dir, 'template.yaml')
        result = self.loader.create_rule_pack_template(template_path)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(template_path))
        
        # Verify template content
        with open(template_path, 'r') as f:
            template = yaml.safe_load(f)
        
        self.assertIn('metadata', template)
        self.assertIn('market_regimes', template)
        self.assertEqual(template['metadata']['name'], 'template_rule_pack')
    
    def test_load_rule_pack(self):
        """Test loading a rule pack from file"""
        # Write test rules to file
        rules_path = os.path.join(self.test_dir, 'test_rules.yaml')
        with open(rules_path, 'w') as f:
            yaml.dump(self.test_rules, f)
        
        # Load the rules
        loaded_rules = self.loader.load_rule_pack('test_rules.yaml')
        
        self.assertEqual(loaded_rules['metadata']['name'], 'test_rules')
        self.assertIn('test_regime', loaded_rules['market_regimes'])
    
    def test_validate_rule_pack(self):
        """Test rule pack validation"""
        # Set up schema
        self.loader.schema = self.test_schema
        
        # Test valid rule pack
        validation_result = self.loader.validate_rule_pack(self.test_rules)
        self.assertTrue(validation_result['valid'])
        self.assertEqual(len(validation_result['errors']), 0)
        
        # Test invalid rule pack (missing required field)
        invalid_rules = self.test_rules.copy()
        del invalid_rules['metadata']['name']
        
        validation_result = self.loader.validate_rule_pack(invalid_rules)
        self.assertFalse(validation_result['valid'])
        self.assertGreater(len(validation_result['errors']), 0)
    
    def test_validate_data_types(self):
        """Test data type validation"""
        self.loader.schema = self.test_schema
        
        # Test valid data types
        validation_result = self.loader.validate_rule_pack(self.test_rules)
        self.assertTrue(validation_result['valid'])
        
        # Test invalid data type
        invalid_rules = self.test_rules.copy()
        invalid_rules['metadata']['version'] = 123  # Should be string
        
        validation_result = self.loader.validate_rule_pack(invalid_rules)
        self.assertFalse(validation_result['valid'])
    
    def test_validate_value_ranges(self):
        """Test value range validation"""
        self.loader.schema = self.test_schema
        
        # Test valid value range
        validation_result = self.loader.validate_rule_pack(self.test_rules)
        self.assertTrue(validation_result['valid'])
        
        # Test invalid value range
        invalid_rules = self.test_rules.copy()
        invalid_rules['market_regimes']['test_regime']['confidence_threshold'] = 1.5  # Should be <= 1.0
        
        validation_result = self.loader.validate_rule_pack(invalid_rules)
        self.assertFalse(validation_result['valid'])
    
    def test_get_rule_pack(self):
        """Test retrieving cached rule packs"""
        # Load a rule pack first
        rules_path = os.path.join(self.test_dir, 'test_rules.yaml')
        with open(rules_path, 'w') as f:
            yaml.dump(self.test_rules, f)
        
        self.loader.load_rule_pack('test_rules.yaml')
        
        # Retrieve from cache
        cached_rules = self.loader.get_rule_pack('test_rules')
        self.assertIsNotNone(cached_rules)
        self.assertEqual(cached_rules['metadata']['name'], 'test_rules')
        
        # Test non-existent rule pack
        non_existent = self.loader.get_rule_pack('non_existent')
        self.assertIsNone(non_existent)
    
    def test_list_rule_packs(self):
        """Test listing available rule packs"""
        # Initially empty
        self.assertEqual(len(self.loader.list_rule_packs()), 0)
        
        # Load a rule pack
        rules_path = os.path.join(self.test_dir, 'test_rules.yaml')
        with open(rules_path, 'w') as f:
            yaml.dump(self.test_rules, f)
        
        self.loader.load_rule_pack('test_rules.yaml')
        
        # Should now have one rule pack
        rule_packs = self.loader.list_rule_packs()
        self.assertEqual(len(rule_packs), 1)
        self.assertIn('test_rules', rule_packs)
    
    def test_get_specific_rules(self):
        """Test getting specific rule categories"""
        # Load test rules
        rules_path = os.path.join(self.test_dir, 'test_rules.yaml')
        with open(rules_path, 'w') as f:
            yaml.dump(self.test_rules, f)
        
        self.loader.load_rule_pack('test_rules.yaml')
        
        # Test getting market regime rules
        market_rules = self.loader.get_market_regime_rules('test_rules')
        self.assertIn('test_regime', market_rules)
        
        # Test getting technical signal rules
        tech_rules = self.loader.get_technical_signal_rules('test_rules')
        self.assertEqual(tech_rules, {})
    
    def test_export_rule_pack(self):
        """Test exporting a rule pack"""
        # Load test rules
        rules_path = os.path.join(self.test_dir, 'test_rules.yaml')
        with open(rules_path, 'w') as f:
            yaml.dump(self.test_rules, f)
        
        self.loader.load_rule_pack('test_rules.yaml')
        
        # Export to new location
        export_path = os.path.join(self.test_dir, 'exported_rules.yaml')
        result = self.loader.export_rule_pack('test_rules', export_path)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(export_path))
        
        # Verify exported content
        with open(export_path, 'r') as f:
            exported = yaml.safe_load(f)
        
        self.assertEqual(exported['metadata']['name'], 'test_rules')
    
    def test_reload_rule_packs(self):
        """Test reloading all rule packs"""
        # Load initial rule pack
        rules_path = os.path.join(self.test_dir, 'test_rules.yaml')
        with open(rules_path, 'w') as f:
            yaml.dump(self.test_rules, f)
        
        self.loader.load_rule_pack('test_rules.yaml')
        self.assertEqual(len(self.loader.list_rule_packs()), 1)
        
        # Reload
        self.loader.reload_rule_packs()
        self.assertEqual(len(self.loader.list_rule_packs()), 1)
        
        # Add new rule pack
        new_rules = self.test_rules.copy()
        new_rules['metadata']['name'] = 'new_rules'
        new_rules_path = os.path.join(self.test_dir, 'new_rules.yaml')
        with open(new_rules_path, 'w') as f:
            yaml.dump(new_rules, f)
        
        # Reload should pick up new file
        self.loader.reload_rule_packs()
        self.assertEqual(len(self.loader.list_rule_packs()), 2)
        self.assertIn('new_rules', self.loader.list_rule_packs())

if __name__ == '__main__':
    unittest.main()
