#!/usr/bin/env python3
"""
Data Splits Management Tool for TTI Recommendation System

This tool helps manage consistent data splits across experiments:
- View current split configuration
- List available splits
- Create new splits with custom parameters
- View split statistics and user group distributions

Usage:
    python manage_splits.py --status
    python manage_splits.py --create --sample-size 100000
    python manage_splits.py --list
    python manage_splits.py --rebuild
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from core.global_data_manager import global_data_manager

def show_status():
    """Show current splits status."""
    print("🔍 Current Data Splits Status")
    print("=" * 50)
    
    try:
        # Check if splits exist
        if global_data_manager.splits_exist():
            global_data_manager.ensure_splits_ready()
            
            summary = global_data_manager.get_splits_summary()
            splits_info = summary['splits_info']
            
            print(f"✅ Splits Status: READY")
            print(f"📁 Path: {summary['path']}")
            print(f"🔐 Signature: {summary['signature']}")
            print(f"🕒 Created: {splits_info['created_at']}")
            print()
            
            print("📊 Dataset Statistics:")
            print(f"   Training samples: {splits_info['train_size']:,}")
            print(f"   Testing samples: {splits_info['test_size']:,}")
            print(f"   Total users: {splits_info['total_users']:,}")
            print(f"   Total movies: {splits_info['total_movies']:,}")
            print()
            
            print("👥 User Group Distribution:")
            for group, info in splits_info['user_groups'].items():
                print(f"   {group:12}: {info['count']:5,} users - {info['description']}")
            print()
            
            print("⚙️  Configuration:")
            config_info = splits_info['config']
            print(f"   Sample size: {config_info.get('SAMPLE_SIZE', 'Full dataset')}")
            print(f"   Positive threshold: {config_info.get('POSITIVE_RATING_THRESHOLD')}")
            print(f"   Test size: {config_info.get('TEST_SIZE')}")
            print(f"   Random state: {config_info.get('RANDOM_STATE')}")
            
            thresholds = config_info.get('USER_GROUP_THRESHOLDS', {})
            print(f"   User thresholds: moderate≤{thresholds.get('moderate_users')}, active≤{thresholds.get('active_users')}")
            
        else:
            print("❌ Splits Status: NOT FOUND")
            print("   No splits exist for current configuration")
            print("   Run with --create to create new splits")
            
    except Exception as e:
        print(f"❌ Error checking splits: {e}")

def list_available_splits():
    """List all available split configurations."""
    print("📋 Available Data Splits")
    print("=" * 50)
    
    available_splits = global_data_manager.list_available_splits()
    
    if not available_splits:
        print("❌ No splits found")
        print("   Run with --create to create new splits")
        return
    
    for i, split in enumerate(available_splits, 1):
        print(f"\n{i}. Split Configuration")
        print(f"   Path: {split['path']}")
        print(f"   Created: {split['created_at']}")
        print(f"   Signature: {split['signature']}")
        print(f"   Train/Test: {split['train_size']:,} / {split['test_size']:,}")
        
        config_info = split['config']
        sample_size = config_info.get('SAMPLE_SIZE', 'Full')
        print(f"   Sample size: {sample_size}")
        user_groups_str = ', '.join([f"{g}({info['count']})" for g, info in split['user_groups'].items()])
        print(f"   User groups: {user_groups_str}")

def create_splits(sample_size=None, force=False):
    """Create new data splits."""
    print("🔧 Creating New Data Splits")
    print("=" * 50)
    
    if sample_size is not None:
        global_data_manager.config['SAMPLE_SIZE'] = sample_size
        print(f"🔧 Using sample size: {sample_size}")
    
    if force:
        global_data_manager.force_rebuild = True
        print("🔧 Force rebuild enabled")
    
    try:
        # Create splits
        global_data_manager.ensure_splits_ready()
        
        print("\n✅ Splits created successfully!")
        
        # Show new splits status
        print("\n" + "="*30)
        show_status()
        
    except Exception as e:
        print(f"❌ Error creating splits: {e}")
        return False
    
    return True

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="TTI Data Splits Management Tool")
    
    # Main actions
    parser.add_argument("--status", action="store_true",
                       help="Show current splits status")
    parser.add_argument("--list", action="store_true", 
                       help="List all available splits")
    parser.add_argument("--create", action="store_true",
                       help="Create new data splits")
    parser.add_argument("--rebuild", action="store_true",
                       help="Rebuild splits (force)")
    
    # Configuration options
    parser.add_argument("--sample-size", type=int,
                       help="Sample size for new splits")
    parser.add_argument("--positive-threshold", type=float,
                       help="Positive rating threshold")
    parser.add_argument("--test-size", type=float, 
                       help="Test set proportion")
    
    args = parser.parse_args()
    
    # Default action if none specified
    if not any([args.status, args.list, args.create, args.rebuild]):
        args.status = True
    
    # Apply configuration overrides
    if args.sample_size is not None:
        global_data_manager.config['SAMPLE_SIZE'] = args.sample_size
    if args.positive_threshold is not None:
        global_data_manager.config['POSITIVE_RATING_THRESHOLD'] = args.positive_threshold
    if args.test_size is not None:
        global_data_manager.config['TEST_SIZE'] = args.test_size
    
    # Execute actions
    if args.status:
        show_status()
    
    elif args.list:
        list_available_splits()
    
    elif args.create:
        success = create_splits(args.sample_size, force=False)
        if not success:
            sys.exit(1)
    
    elif args.rebuild:
        print("⚠️  Rebuilding will delete existing splits for this configuration!")
        response = input("Continue? (y/N): ")
        if response.lower() == 'y':
            success = create_splits(args.sample_size, force=True)
            if not success:
                sys.exit(1)
        else:
            print("❌ Rebuild cancelled")

if __name__ == "__main__":
    main() 