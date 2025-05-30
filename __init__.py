from .leffaNode import CXH_Leffa_Viton_Load, CXH_Leffa_Viton_Run, CXH_Leffa_Mask_Generator, CXH_Leffa_Pose_Preprocessor

# NEW MASKING NODES - Using the latest mask/ folder logic
try:
    from .newMaskingNodes import CXH_NEW_Advanced_Mask_Generator, CXH_NEW_Advanced_Pose_Preprocessor
    NEW_NODES_AVAILABLE = True
    print("🚀 NEW MASKING NODES IMPORTED SUCCESSFULLY!")
except ImportError as e:
    NEW_NODES_AVAILABLE = False
    print(f"❌ New masking nodes not available: {e}")
    print("📝 Using legacy nodes only")

NODE_CLASS_MAPPINGS = {
    "CXH_Leffa_Viton_Load": CXH_Leffa_Viton_Load,
    "CXH_Leffa_Viton_Run": CXH_Leffa_Viton_Run,
    "CXH_Leffa_Mask_Generator": CXH_Leffa_Mask_Generator,  # OLD - DEPRECATED
    "CXH_Leffa_Pose_Preprocessor": CXH_Leffa_Pose_Preprocessor,  # OLD - DEPRECATED
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CXH_Leffa_Viton_Load": "CXH_Leffa_Viton_Load",
    "CXH_Leffa_Viton_Run": "CXH_Leffa_Viton_Run",
    "CXH_Leffa_Mask_Generator": "CXH_Leffa_Mask_Generator [DEPRECATED]",  # OLD
    "CXH_Leffa_Pose_Preprocessor": "CXH_Leffa_Pose_Preprocessor [DEPRECATED]",  # OLD
}

# Add new nodes if available
if NEW_NODES_AVAILABLE:
    NODE_CLASS_MAPPINGS.update({
        "CXH_NEW_Advanced_Mask_Generator": CXH_NEW_Advanced_Mask_Generator,  # NEW
        "CXH_NEW_Advanced_Pose_Preprocessor": CXH_NEW_Advanced_Pose_Preprocessor,  # NEW
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "CXH_NEW_Advanced_Mask_Generator": "🚀 CXH NEW Advanced Mask Generator",  # NEW
        "CXH_NEW_Advanced_Pose_Preprocessor": "🚀 CXH NEW Advanced Pose Preprocessor",  # NEW
    })
    print("🚀 NEW ADVANCED MASKING NODES REGISTERED SUCCESSFULLY!")