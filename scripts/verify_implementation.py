"""
Verification script to check model implementation without running it.
This script validates the code structure and imports.
"""

import ast
import sys
from pathlib import Path

def check_file_syntax(file_path: Path) -> bool:
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        print(f"✓ {file_path.name}: Syntax valid")
        return True
    except SyntaxError as e:
        print(f"✗ {file_path.name}: Syntax error at line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"✗ {file_path.name}: Error: {e}")
        return False

def check_class_exists(file_path: Path, class_name: str) -> bool:
    """Check if a class exists in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                print(f"✓ {file_path.name}: Found class '{class_name}'")
                return True
        
        print(f"✗ {file_path.name}: Class '{class_name}' not found")
        return False
    except Exception as e:
        print(f"✗ {file_path.name}: Error checking class: {e}")
        return False

def check_function_exists(file_path: Path, function_name: str) -> bool:
    """Check if a function exists in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                print(f"✓ {file_path.name}: Found function '{function_name}'")
                return True
        
        print(f"✗ {file_path.name}: Function '{function_name}' not found")
        return False
    except Exception as e:
        print(f"✗ {file_path.name}: Error checking function: {e}")
        return False

def main():
    """Run verification checks."""
    print("="*70)
    print("Optimized-CodeBERT Implementation Verification")
    print("="*70)
    print()
    
    project_root = Path(__file__).parent.parent
    src_ml = project_root / "src" / "ml"
    
    all_passed = True
    
    # Check dataset.py
    print("Checking dataset.py...")
    dataset_file = src_ml / "dataset.py"
    if dataset_file.exists():
        all_passed &= check_file_syntax(dataset_file)
        all_passed &= check_class_exists(dataset_file, "VulnerabilityDataset")
        all_passed &= check_function_exists(dataset_file, "split_dataset")
    else:
        print(f"✗ File not found: {dataset_file}")
        all_passed = False
    print()
    
    # Check models.py
    print("Checking models.py...")
    models_file = src_ml / "models.py"
    if models_file.exists():
        all_passed &= check_file_syntax(models_file)
        all_passed &= check_class_exists(models_file, "OptimizedCodeBERT")
        all_passed &= check_class_exists(models_file, "OptimizedCodeBERTWithWeightedLoss")
        all_passed &= check_function_exists(models_file, "create_model")
    else:
        print(f"✗ File not found: {models_file}")
        all_passed = False
    print()
    
    # Check train.py
    print("Checking train.py...")
    train_file = src_ml / "train.py"
    if train_file.exists():
        all_passed &= check_file_syntax(train_file)
        all_passed &= check_class_exists(train_file, "Trainer")
        all_passed &= check_function_exists(train_file, "evaluate_model")
    else:
        print(f"✗ File not found: {train_file}")
        all_passed = False
    print()
    
    # Check train_baseline.py
    print("Checking train_baseline.py...")
    train_script = project_root / "scripts" / "train_baseline.py"
    if train_script.exists():
        all_passed &= check_file_syntax(train_script)
        all_passed &= check_function_exists(train_script, "main")
    else:
        print(f"✗ File not found: {train_script}")
        all_passed = False
    print()
    
    # Check configuration files
    print("Checking configuration files...")
    configs_dir = project_root / "configs"
    
    labels_file = configs_dir / "labels.yaml"
    if labels_file.exists():
        print(f"✓ labels.yaml exists")
        # Check if "safe" label is present
        with open(labels_file, 'r') as f:
            content = f.read()
            if "safe:" in content:
                print(f"✓ labels.yaml contains 'safe' label")
            else:
                print(f"✗ labels.yaml missing 'safe' label")
                all_passed = False
    else:
        print(f"✗ File not found: {labels_file}")
        all_passed = False
    
    train_config = configs_dir / "train.yaml"
    if train_config.exists():
        print(f"✓ train.yaml exists")
        # Check if model config is present
        with open(train_config, 'r') as f:
            content = f.read()
            if "model:" in content and "training:" in content:
                print(f"✓ train.yaml contains model and training config")
            else:
                print(f"✗ train.yaml missing required sections")
                all_passed = False
    else:
        print(f"✗ File not found: {train_config}")
        all_passed = False
    print()
    
    # Check requirements.txt
    print("Checking requirements.txt...")
    req_file = project_root / "requirements.txt"
    if req_file.exists():
        with open(req_file, 'r') as f:
            content = f.read()
            required_packages = ['torch', 'transformers', 'scikit-learn', 'tensorboard']
            missing = []
            for pkg in required_packages:
                if pkg not in content:
                    missing.append(pkg)
            
            if not missing:
                print(f"✓ requirements.txt contains all required packages")
            else:
                print(f"✗ requirements.txt missing: {', '.join(missing)}")
                all_passed = False
    else:
        print(f"✗ File not found: {req_file}")
        all_passed = False
    print()
    
    # Summary
    print("="*70)
    if all_passed:
        print("✓ All verification checks passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Prepare data: python scripts/run_hf_pipeline.py")
        print("3. Train model: python scripts/train_baseline.py")
    else:
        print("✗ Some verification checks failed. Please review the errors above.")
        sys.exit(1)
    print("="*70)

if __name__ == "__main__":
    main()
