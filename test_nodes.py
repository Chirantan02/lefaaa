#!/usr/bin/env python3
"""
Test script to verify that all nodes are properly loaded and available.
"""

def test_node_imports():
    """Test that all nodes can be imported successfully."""
    print("🧪 Testing node imports...")
    
    try:
        # Test legacy nodes
        from leffaNode import CXH_Leffa_Viton_Load, CXH_Leffa_Viton_Run, CXH_Leffa_Mask_Generator, CXH_Leffa_Pose_Preprocessor
        print("✅ Legacy nodes imported successfully")
        
        # Test new nodes
        from newMaskingNodes import CXH_NEW_Advanced_Mask_Generator, CXH_NEW_Advanced_Pose_Preprocessor
        print("✅ New masking nodes imported successfully")
        
        # Test node class mappings
        from __init__ import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        print("✅ Node mappings imported successfully")
        
        print(f"\n📊 Available nodes:")
        for node_name, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
            print(f"  - {node_name}: {display_name}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_node_input_types():
    """Test that nodes have proper INPUT_TYPES defined."""
    print("\n🧪 Testing node INPUT_TYPES...")
    
    try:
        from newMaskingNodes import CXH_NEW_Advanced_Mask_Generator, CXH_NEW_Advanced_Pose_Preprocessor
        
        # Test mask generator
        mask_inputs = CXH_NEW_Advanced_Mask_Generator.INPUT_TYPES()
        print("✅ CXH_NEW_Advanced_Mask_Generator INPUT_TYPES OK")
        print(f"   Required inputs: {list(mask_inputs['required'].keys())}")
        
        # Test pose preprocessor
        pose_inputs = CXH_NEW_Advanced_Pose_Preprocessor.INPUT_TYPES()
        print("✅ CXH_NEW_Advanced_Pose_Preprocessor INPUT_TYPES OK")
        print(f"   Required inputs: {list(pose_inputs['required'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing INPUT_TYPES: {e}")
        return False

if __name__ == "__main__":
    print("🚀 COMFYUI LEFFA NODE TESTING")
    print("=" * 50)
    
    success = True
    success &= test_node_imports()
    success &= test_node_input_types()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ ALL TESTS PASSED! Nodes are ready to use.")
    else:
        print("❌ SOME TESTS FAILED! Check the errors above.")
    
    print("\n📝 Next steps:")
    print("1. Load the workflow.json in ComfyUI")
    print("2. The new nodes should appear in the node menu")
    print("3. Use CXH_NEW_Advanced_Mask_Generator for masking")
    print("4. Use CXH_NEW_Advanced_Pose_Preprocessor for pose detection")
