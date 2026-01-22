"""
PERSIST (Persistence-driven Spatial Identification of Stability Topologies)
===========================================================================

A topological data analysis pipeline for spatial transcriptomics that combines:
1. Persistent homology (α-complexes) for scale selection
2. Multiscale total variation for boundary detection
3. Watershed segmentation for domain identification

Key Insight: While spatial geometry determines baseline topology, the coupling between
persistent homology features and transcriptional variation reveals tissue-specific
functional organization.

Author: Gaurav Khanal
Date: January 2026
License: MIT

Usage:
    python persist_pipeline.py --tissue_type human_lymph_node --n_scales 5 --n_top_genes 2000
"""

import argparse
import json
import logging
import os
import sys
import dask
import warnings
from typing import Any, Dict, List, Optional, Tuple

import gudhi as gd
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata, NearestNDInterpolator
from scipy.sparse.csgraph import connected_components
from scipy.stats import entropy, spearmanr
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from tqdm.auto import tqdm


# ============================================================================
# DEPENDENCY MANAGEMENT WITH GRACEFUL FALLBACKS
# ============================================================================

# Gudhi persistence landscapes for topological feature visualization
try:
    from gudhi.representations import Landscape
    LANDSCAPE_AVAILABLE: bool = True
    LOGGING_MESSAGES: Dict[str, str] = {
        "landscape_success": "Gudhi Landscape available for persistence visualization."
    }
except ImportError:
    LANDSCAPE_AVAILABLE = False
    print("WARNING: Gudhi Landscape not available. Persistence landscape plots disabled.")

# Squidpy for spatial autocorrelation analysis
try:
    import squidpy as sq
    SQUIDPY_AVAILABLE: bool = True
    LOGGING_MESSAGES["squidpy_success"] = "Squidpy available for spatial autocorrelation analysis."
except ImportError:
    SQUIDPY_AVAILABLE = False
    print("WARNING: Squidpy not available. Falling back to HVGs for gene selection.")

# GPU acceleration with CuPy (optional but recommended for large datasets)
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    from cupyx import cusparse
    cp.cuda.runtime.getDeviceCount()
    GPU_AVAILABLE: bool = True
    print("INFO: GPU acceleration available via CuPy.")
except (ImportError, OSError, Exception):
    GPU_AVAILABLE = False
    print("INFO: GPU not available. Running on CPU.")

# Plotly for interactive visualizations
try:
    import plotly.express as px
    PLOTLY_AVAILABLE: bool = True
    LOGGING_MESSAGES["plotly_success"] = "Plotly available for interactive visualizations."
except ImportError:
    PLOTLY_AVAILABLE = False
    print("INFO: Plotly not available. Using static matplotlib plots only.")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logger(output_dir: str, log_level: str = "INFO") -> logging.Logger:
    """
    Configure comprehensive logging for the PERSIST pipeline.
    
    Creates both console and file handlers with detailed formatting.
    
    Args:
        output_dir: Directory where log file will be saved
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        logging.Logger: Configured logger instance
        
    Raises:
        PermissionError: If unable to write to log file
        ValueError: If invalid log level specified
    """
    # Validate log level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_levels:
        raise ValueError(f"Invalid log level: {log_level}. Must be one of {valid_levels}")
    
    logger = logging.getLogger("PERSIST")
    
    # Convert string level to logging constant
    level = getattr(logging, log_level)
    logger.setLevel(level)
    
    # Clear any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Detailed format for log messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler for real-time output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # File handler for persistent logging
    try:
        log_file = os.path.join(output_dir, "persist_pipeline.log")
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        logger.info(f"Log file created: {log_file}")
    except PermissionError as e:
        logger.error(f"Cannot write log file: {e}")
        raise
    
    return logger


class PERSISTConfig:
    """
    Configuration parameters for the PERSIST pipeline.
    
    Tuning these parameters can significantly affect results. Default values
    are optimized for 10x Visium datasets but should be adjusted for other
    spatial transcriptomics platforms.
    
    Attributes:
        pruning_neighbors (int): Number of neighbors for initial graph pruning
        floor_multiplier (float): Lower bound for α as a multiple of median nearest-neighbor distance.
        alpha_multiplier (float): Upper bound for α as a multiple of median nearest-neighbor distance.
        min_scales (int): Minimum number of scales for multiscale analysis
        watershed_sigma (float): Gaussian filter sigma for watershed preprocessing
        watershed_threshold_abs (float): Absolute threshold for peak detection in watershed 
        watershed_mask_threshold (float): Minimum value for watershed mask
        moran_perms (int): Number of permutations for Moran's I test
        bootstrap_runs (int): Number of bootstrap iterations for robustness assessment
        bootstrap_sample_frac (float): Fraction of samples to use in each bootstrap run
        bootstrap_significance (float): Threshold for declaring a core as significant
        grad_clip_percentiles (Tuple[float, float]): Percentiles for clipping gradient magnitudes
        frontier_clip_percentile (float): Percentile for clipping frontier TV magnitude
        random_seed (int): Random seed for reproducibility
        visualize_mesh (bool): Whether to generate mesh visualization
        max_grid_dim (int): Maximum dimension for watershed grid (for memory efficiency)
        degree_normalize_tv (bool): Whether to normalize TV by vertex degree
    """
    
    def __init__(
        self,
        pruning_neighbors: int = 3,
        floor_multiplier: float = 1.1,
        alpha_multiplier: float = 2.0,
        min_scales: int = 5,
        watershed_sigma: float = 0.8,
        watershed_threshold_abs: float = 0.4, # Suppress weak maxima
        watershed_mask_threshold: float = 0.1, # Ignore low-confidence background
        moran_perms: int = 1000,
        bootstrap_runs: int = 100,
        bootstrap_sample_frac: float = 0.85,
        bootstrap_significance: float = 0.7,
        grad_clip_percentiles: Tuple[float, float] = (5, 95),
        frontier_clip_percentile: float = 99,
        random_seed: int = 42,
        visualize_mesh: bool = True,
        max_grid_dim: int = 500,
        degree_normalize_tv: bool = True,
        tissue_type: str = "breast_cancer"
    ) -> None:
        """
        Initialize PERSIST configuration with validated parameters.
        
        Args:
            pruning_neighbors: Number of neighbors for initial graph pruning
            floor_multiplier: Minimum α-complex scale as multiple of median distance
            alpha_multiplier: Maximum α-complex scale as multiple of median distance
            min_scales: Minimum number of scales for multiscale analysis
            watershed_sigma: Gaussian filter sigma for watershed preprocessing
            watershed_threshold_abs: Absolute threshold for peak detection in watershed
            watershed_mask_threshold: Minimum value for watershed mask
            moran_perms: Number of permutations for Moran's I test
            bootstrap_runs: Number of bootstrap iterations for robustness assessment
            bootstrap_sample_frac: Fraction of samples to use in each bootstrap run
            bootstrap_significance: Threshold for declaring a core as significant
            grad_clip_percentiles: Percentiles for clipping gradient magnitudes
            frontier_clip_percentile: Percentile for clipping frontier TV magnitude
            random_seed: Random seed for reproducibility
            visualize_mesh: Whether to generate mesh visualization
            max_grid_dim: Maximum dimension for watershed grid (for memory efficiency)
            degree_normalize_tv: Whether to normalize TV by vertex degree
            
        Raises:
            ValueError: If any parameter has an invalid value
        """
        # Validate numeric parameters
        if pruning_neighbors < 1:
            raise ValueError("pruning_neighbors must be at least 1")
        if floor_multiplier <= 0:
            raise ValueError("floor_multiplier must be positive")
        if alpha_multiplier <= floor_multiplier:
            raise ValueError("alpha_multiplier must be greater than floor_multiplier")
        if min_scales < 1:
            raise ValueError("min_scales must be at least 1")
        if watershed_sigma < 0:
            raise ValueError("watershed_sigma must be non-negative")
        if not 0 <= watershed_threshold_abs <= 1:
            raise ValueError("watershed_threshold_abs must be between 0 and 1")
        if not 0 <= watershed_mask_threshold <= 1:
            raise ValueError("watershed_mask_threshold must be between 0 and 1")
        if moran_perms < 10:
            raise ValueError("moran_perms must be at least 10 for statistical validity")
        if bootstrap_runs < 5:
            raise ValueError("bootstrap_runs must be at least 5")
        if not 0 < bootstrap_sample_frac <= 1:
            raise ValueError("bootstrap_sample_frac must be between 0 and 1")
        if not 0 <= bootstrap_significance <= 1:
            raise ValueError("bootstrap_significance must be between 0 and 1")
        if not 0 <= grad_clip_percentiles[0] < grad_clip_percentiles[1] <= 100:
            raise ValueError("grad_clip_percentiles must be ascending values between 0 and 100")
        if not 0 <= frontier_clip_percentile <= 100:
            raise ValueError("frontier_clip_percentile must be between 0 and 100")
        if max_grid_dim < 50:
            raise ValueError("max_grid_dim must be at least 50 for meaningful resolution")
        
        # Assign validated parameters
        self.pruning_neighbors: int = pruning_neighbors
        self.floor_multiplier: float = floor_multiplier
        self.alpha_multiplier: float = alpha_multiplier
        self.min_scales: int = min_scales
        self.watershed_sigma: float = watershed_sigma
        self.watershed_threshold_abs: float = watershed_threshold_abs
        self.watershed_mask_threshold: float = watershed_mask_threshold
        self.moran_perms: int = moran_perms
        self.bootstrap_runs: int = bootstrap_runs
        self.bootstrap_sample_frac: float = bootstrap_sample_frac
        self.bootstrap_significance: float = bootstrap_significance
        self.grad_clip_percentiles: Tuple[float, float] = grad_clip_percentiles
        self.frontier_clip_percentile: float = frontier_clip_percentile
        self.random_seed: int = random_seed
        self.visualize_mesh: bool = visualize_mesh
        self.max_grid_dim: int = max_grid_dim
        self.degree_normalize_tv: bool = degree_normalize_tv
        self.tissue_type = tissue_type
        
        # Load tissue schema
        self._load_tissue_schema()
        
        # Log configuration summary
        self._log_configuration()
    
    def _log_configuration(self) -> None:
        """Log the configuration parameters for transparency."""
        config_summary = (
            f"PERSIST Configuration:\n"
            f"  - Pruning neighbors: {self.pruning_neighbors}\n"
            f"  - Scale range: {self.floor_multiplier:.1f}-{self.alpha_multiplier:.1f}x median distance\n"
            f"  - Min scales: {self.min_scales}\n"
            f"  - Watershed sigma: {self.watershed_sigma}\n"
            f"  - Bootstrap runs: {self.bootstrap_runs}\n"
            f"  - Random seed: {self.random_seed}"
        )
        print(f"INFO: {config_summary}")

    def _load_tissue_schema(self):
        """Injects biological knowledge based on tissue_type."""
        if self.tissue_type in TISSUE_SCHEMA:
            self.tissue_config = TISSUE_SCHEMA[self.tissue_type]
        else:
            logging.warning(f"Unknown tissue '{self.tissue_type}'. Using defaults.")
            self.tissue_config = {}

        # --- LOAD MARKERS (Fixes the plotting error) ---
        self.stability_markers = self.tissue_config.get("stability_markers", {})
        self.reference_markers = self.tissue_config.get("reference_markers", [])

        # --- LOAD MESH STRATEGY ---
        mesh_conf = self.tissue_config.get("mesh_config", {})
        self.mesh_strategy = mesh_conf.get("strategy", "largest_component")
        self.min_k = mesh_conf.get("min_k", 3)
        self.auto_bridge = mesh_conf.get("auto_bridge", False)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of configuration
        """
        return {
            'pruning_neighbors': self.pruning_neighbors,
            'floor_multiplier': self.floor_multiplier,
            'alpha_multiplier': self.alpha_multiplier,
            'min_scales': self.min_scales,
            'watershed_sigma': self.watershed_sigma,
            'watershed_threshold_abs': self.watershed_threshold_abs,
            'watershed_mask_threshold': self.watershed_mask_threshold,
            'moran_perms': self.moran_perms,
            'bootstrap_runs': self.bootstrap_runs,
            'bootstrap_sample_frac': self.bootstrap_sample_frac,
            'bootstrap_significance': self.bootstrap_significance,
            'grad_clip_percentiles': self.grad_clip_percentiles,
            'frontier_clip_percentile': self.frontier_clip_percentile,
            'random_seed': self.random_seed,
            'visualize_mesh': self.visualize_mesh,
            'max_grid_dim': self.max_grid_dim,
            'degree_normalize_tv': self.degree_normalize_tv

        }


# ============================================================================
# BIOLOGICAL KNOWLEDGE BASE
# ============================================================================

TISSUE_SCHEMA: Dict[str, Dict[str, Any]] = {
    # ========================================================================
    # HUMAN BREAST CANCER (10x Visium)
    # ========================================================================
    "breast_cancer": {
        "description": "Ductal carcinoma with complex tumor-stroma interfaces",
        
        # --- TOPOLOGICAL CONFIGURATION ---
        "mesh_config": {
            "strategy": "multi_component",    # Keep tumor islands
            "min_k": 3,                       # Keep local connectivity tight
            "auto_bridge": True               # Bridge fatty tissue gaps
        },

        # --- BIOLOGICAL MARKERS (High Resolution) ---
        "stability_markers": {
            # THE CORE: Duct structures should be highly stable
            "Tumor_Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1"],
            
            # THE BOUNDARY: Myoepithelial cells often form the "walls" of ducts
            "Myoepithelial (Walls)": ["KRT14", "ACTA2", "MYLK", "TP63"],
            
            # THE ACTIVITY: Proliferation usually happens in stable cores
            "Proliferation": ["MKI67", "TOP2A", "PCNA", "CENPF"],
            
            # THE ARCHITECTURE: Fibroblasts define the "highway" (Unstable/Transitional)
            "Fibroblast_Stroma": ["COL1A1", "COL1A2", "FN1", "DCN", "LUM"],
            
            # THE VASCULATURE: Endothelial cells (Transitional)
            "Endothelial": ["PECAM1", "VWF", "CD34", "ENG"],
            
            # THE INFILTRATE: Immune cells (Can be stable aggregates or diffuse)
            "T_Cells": ["CD3D", "CD3E", "CD2"],
            "B_Cells": ["MS4A1", "CD79A", "CD19"],
            "Macrophages": ["CD68", "CD163", "TYROBP"]
        },
        
        # Markers expected to align with High TV (Gradients/Walls)
        "reference_markers": ["ACTA2", "COL1A1", "VIM", "SPARC"] 
    },

    # ========================================================================
    # HUMAN LYMPH NODE
    # ========================================================================
    "human_lymph_node": {
        "description": "Structured immune tissue (Follicles & Germinal Centers)",
        
        "mesh_config": {
            "strategy": "largest_component",
            "min_k": 3,
            "auto_bridge": False
        },

        "stability_markers": {
            # Germinal Centers are the definition of "Stable Cores"
            "Germinal_Center_Dark": ["AICDA", "MKI67", "CXCR4"], 
            "Germinal_Center_Light": ["CD83", "BCL2A1", "NFKBIA"],
            
            # The Mantle Zone forms a ring (Boundary)
            "B_Cell_Mantle": ["CD79A", "MS4A1", "TCL1A", "CD69"],
            
            # T-Zone (Transitional/Diffuse)
            "T_Zone_Parafollicular": ["CD3E", "CD4", "IL7R", "CCL21"],
            
            # Structural/Vascular
            "High_Endothelial_Venules": ["ACKR1", "CHST4", "SELL"]
        },
        "reference_markers": ["CCL19", "CCL21", "CXCL13"] # Chemokine gradients define structure here
    },

    # ========================================================================
    # MOUSE BRAIN (Sagittal)
    # ========================================================================
    "mouse_brain": {
        "description": "Layered neural architecture",
        
        "mesh_config": {
            "strategy": "largest_component",
            "min_k": 3,
            "auto_bridge": False
        },

        "stability_markers": {
            # Hippocampus (The most distinct topological loop in the brain)
            "Hippocampus_CA1": ["Fibcd1", "Wfs1"],
            "Hippocampus_CA3": ["Cck", "Enpp2"],
            "Dentate_Gyrus": ["Prox1", "Dsp"],
            
            # Cortex Layers (Laminar structure)
            "Cortex_L2/3": ["Cux2", "Lamp5"],
            "Cortex_L4": ["Rorb", "Whrn"],
            "Cortex_L5": ["Fezf2", "Bcl11b"],
            "Cortex_L6": ["Foxp2", "Syt6"],
            
            # White Matter (Tracks = Stable Highways)
            "Oligodendrocytes": ["Mbp", "Plp1", "Mobp"],
            
            # Striatum
            "Striatum": ["Penk", "Tac1", "Gpr88"]
        },
        "reference_markers": ["Mbp", "Plp1"] # White matter boundaries
    }
}


# ============================================================================
# CORE MODULES
# ============================================================================

class TissueAdaptiveGeneSelector:
    """
    Tissue-aware gene selection with adaptive thresholds.
    
    Strategy:
    1. Pre-filter by variance (computational efficiency)
    2. Assess spatial autocorrelation (Moran's I)
    3. Apply tissue-specific adaptive thresholds
    4. Ensure biological validity (min/max gene counts)
    
    The key innovation: thresholds and strategies adapt to tissue biology.
    """
    
    # Tissue-specific selection profiles
    TISSUE_PROFILES = {
        "human_lymph_node": {
            "description": "Discrete immune compartments with sharp boundaries",
            "pre_filter_n": 5000,
            "min_moran_base": 0.20,      # High threshold - expect clear spatial structure
            "moran_percentile": 70,       # Top 30% of spatially variable genes
            "adaptive_strategy": "strict",
            "expected_patterns": "discrete_compartments"
        },
        "mouse_brain": {
            "description": "Layered neural tissue with continuous gradients",
            "pre_filter_n": 8000,         # More genes - complex tissue
            "min_moran_base": 0.15,       # Moderate threshold - gradual transitions
            "moran_percentile": 65,       # Top 35% of spatially variable genes
            "adaptive_strategy": "balanced",
            "expected_patterns": "continuous_gradients"
        },
        "breast_cancer": {
            "description": "Heterogeneous tumor with fragmented microenvironments",
            "pre_filter_n": 4000,
            "min_moran_base": 0.25,       # Higher threshold - filter out noise
            "moran_percentile": 75,       # Top 25% - focus on strong patterns
            "adaptive_strategy": "robust",  # Tolerate fragmentation
            "expected_patterns": "fragmented_patches"
        }
    }
    
    def __init__(
        self,
        adata: sc.AnnData,
        logger: logging.Logger,
        target_genes: int = 2000,
        max_genes: int = 5000
    ):
        """
        Initialize tissue-adaptive gene selector.
        
        Args:
            adata: AnnData object with expression data
            logger: Logger instance
            target_genes: Minimum genes to select (safety floor)
            max_genes: Maximum genes to select (computational ceiling)
        """
        self.adata = adata
        self.logger = logger
        self.target_genes = target_genes
        self.max_genes = max_genes
        
        # Will be populated during selection
        self.tissue_profile = None
        self.moran_results = None
    
    def select_genes(
        self,
        tissue_type: str,
        force_moran_threshold: Optional[float] = None,
        force_n_prefilter: Optional[int] = None
    ) -> sc.AnnData:
        """
        Main entry point: tissue-adaptive gene selection.
        
        Args:
            tissue_type: One of "human_lymph_node", "mouse_brain", "breast_cancer"
            force_moran_threshold: Override automatic threshold (for experimentation)
            force_n_prefilter: Override pre-filter gene count
            
        Returns:
            AnnData subset with selected genes
        """
        # Step 1: Load tissue profile
        self.tissue_profile = self._get_tissue_profile(tissue_type)
        
        # Apply manual overrides if provided
        if force_moran_threshold is not None:
            self.logger.info(f"Overriding Moran's I threshold: {force_moran_threshold}")
            self.tissue_profile['min_moran_base'] = force_moran_threshold
        
        if force_n_prefilter is not None:
            self.logger.info(f"Overriding pre-filter count: {force_n_prefilter}")
            self.tissue_profile['pre_filter_n'] = force_n_prefilter
        
        self._log_selection_strategy()
        
        # Step 2: Variance pre-filter (fast)
        adata_hvg = self._prefilter_by_variance()
        
        # Step 3: Spatial autocorrelation filter (adaptive)
        if SQUIDPY_AVAILABLE:
            try:
                selected_genes = self._adaptive_spatial_selection(adata_hvg)
                return self.adata[:, selected_genes].copy()
            
            except Exception as e:
                self.logger.error(
                    f"Spatial autocorrelation failed: {e}. "
                    f"Falling back to variance-only selection."
                )
        else:
            self.logger.warning(
                "Squidpy not available. Using variance-only selection."
            )
        
        # Step 4: Fallback (variance only)
        return self._fallback_variance_selection()
    
    def _get_tissue_profile(self, tissue_type: str) -> Dict[str, Any]:
        """
        Load tissue-specific selection profile.
        
        Returns validated profile with all required parameters.
        """
        if tissue_type not in self.TISSUE_PROFILES:
            self.logger.warning(
                f"Unknown tissue type '{tissue_type}'. "
                f"Available: {list(self.TISSUE_PROFILES.keys())}. "
                f"Using 'mouse_brain' defaults."
            )
            tissue_type = "mouse_brain"
        
        profile = self.TISSUE_PROFILES[tissue_type].copy()
        
        # Ensure pre_filter_n doesn't exceed total genes
        profile['pre_filter_n'] = min(
            profile['pre_filter_n'],
            self.adata.shape[1]
        )
        
        # Ensure it's at least target_genes
        profile['pre_filter_n'] = max(
            profile['pre_filter_n'],
            self.target_genes
        )
        
        return profile
    
    def _log_selection_strategy(self) -> None:
        """Log the adaptive selection strategy."""
        p = self.tissue_profile
        
        strategy_msg = f"""
        ╔════════════════════════════════════════════════════════╗
        ║     TISSUE-ADAPTIVE GENE SELECTION STRATEGY            ║
        ╠════════════════════════════════════════════════════════╣
        ║  Tissue Pattern: {p['expected_patterns']:<30} ║
        ║  Strategy: {p['adaptive_strategy']:<42} ║
        ║  Pre-filter: {p['pre_filter_n']:<44} ║
        ║  Moran's I Base: {p['min_moran_base']:<38.2f} ║
        ║  Percentile Cutoff: {p['moran_percentile']:<33}th ║
        ║  Target Range: {self.target_genes}-{self.max_genes} genes{' ' * 25} ║
        ╚════════════════════════════════════════════════════════╝
        """
        
        self.logger.info(strategy_msg)
    
    def _prefilter_by_variance(self) -> sc.AnnData:
        """
        Stage 1: Fast pre-filter by variance.
        
        This removes obviously uninformative genes before expensive
        spatial autocorrelation computation.
        """
        n_prefilter = self.tissue_profile['pre_filter_n']
        
        self.logger.info(
            f"Stage 1: Variance pre-filter (top {n_prefilter} HVGs)..."
        )
        
        try:
            sc.pp.highly_variable_genes(self.adata, n_top_genes=n_prefilter)
            adata_hvg = self.adata[:, self.adata.var['highly_variable']].copy()
            
            self.logger.info(
                f"  → Pre-filtered: {self.adata.shape[1]} → {adata_hvg.shape[1]} genes"
            )
            
            return adata_hvg
            
        except Exception as e:
            raise RuntimeError(f"Variance pre-filtering failed: {e}")
    
    def _adaptive_spatial_selection(self, adata_hvg: sc.AnnData) -> List[str]:
        """
        Stage 2: Adaptive spatial autocorrelation selection.
        
        The key innovation: adapts to tissue-specific spatial patterns.
        
        Three adaptive strategies:
        - 'strict': High Moran's I threshold, discrete compartments expected
        - 'balanced': Moderate threshold, gradients expected  
        - 'robust': Percentile-based, handles fragmentation
        """
        self.logger.info("Stage 2: Spatial autocorrelation analysis...")
        
        # Compute Moran's I
        self.moran_results = self._compute_morans_i(adata_hvg)
        
        # Assess spatial quality
        spatial_quality = self._assess_spatial_quality(self.moran_results)
        
        # Apply adaptive threshold
        strategy = self.tissue_profile['adaptive_strategy']
        
        if strategy == "strict":
            selected = self._strict_threshold_selection(spatial_quality)
        elif strategy == "balanced":
            selected = self._balanced_threshold_selection(spatial_quality)
        elif strategy == "robust":
            selected = self._robust_percentile_selection(spatial_quality)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return selected
    
    def _compute_morans_i(self, adata_hvg: sc.AnnData) -> pd.DataFrame:
        """
        Compute Moran's I with adaptive parallelism.
        
        Returns:
            DataFrame with Moran's I scores for each gene
        """
        
        # Build spatial graph
        self.logger.info("  → Building spatial neighbor graph...")
        sq.gr.spatial_neighbors(adata_hvg, coord_type="generic")
        
        # Determine parallelism
        n_genes = adata_hvg.shape[1]
        
        if n_genes > 15000:
            n_jobs = self._determine_n_jobs(n_genes)
        else:
            n_jobs = 1
            self.logger.info(f"  → Using serial execution (n_jobs=1) for stability on {n_genes} genes.")

        self.logger.info(
            f"  → Computing Moran's I for {n_genes} genes (n_jobs={n_jobs})..."
        )

        # Compute with retry logic
        try:
            sq.gr.spatial_autocorr(
                adata_hvg,
                mode="moran",
                n_perms=1000,
                n_jobs=n_jobs,
                show_progress_bar=True
            )
        except Exception as e:
            self.logger.warning(
                f"Parallel Moran's I failed: {e}. Retrying serially..."
            )
            sq.gr.spatial_autocorr(
                adata_hvg,
                mode="moran",
                n_perms=1000,
                n_jobs=1,
                show_progress_bar=True
            )
        
        # Validate results
        if 'moranI' not in adata_hvg.uns:
            raise RuntimeError("Moran's I computation failed silently")
        
        moran_df = adata_hvg.uns['moranI']
        
        if moran_df.empty:
            raise RuntimeError("Moran's I returned empty results")
        
        # Log distribution
        self.logger.info(
            f"  → Moran's I distribution: "
            f"max={moran_df['I'].max():.3f}, "
            f"median={moran_df['I'].median():.3f}, "
            f"mean={moran_df['I'].mean():.3f}, "
            f"min={moran_df['I'].min():.3f}"
        )
        
        return moran_df
    
    def _determine_n_jobs(self, n_genes: int) -> int:
        """
        Adaptive parallelism based on gene count and memory.
        
        Strategy:
        - Small datasets (<2000 genes): 2 cores
        - Medium datasets (2000-5000): 4 cores
        - Large datasets (>5000): limit to 4 cores to avoid OOM
        """
        
        if n_genes < 2000:
            n_jobs = 2
        elif n_genes < 5000:
            n_jobs = 4
        else:
            n_jobs = min(4, max(1, os.cpu_count() // 2))
        
        # Estimate memory usage
        n_spots = self.adata.shape[0]
        est_memory_gb = (n_spots * n_genes * 8 * n_jobs * 1.5) / (1024**3)
        
        # ALWAYS log the estimate
        self.logger.info(
            f"  → Memory estimate: ~{est_memory_gb:.1f} GB for n_jobs={n_jobs}"
        )
        
        # Reduce parallelism if high memory usage expected
        if est_memory_gb > 32:
            self.logger.warning("High memory usage expected. Reducing parallelism.")
            n_jobs = max(1, n_jobs // 2)
            est_memory_gb = (n_spots * n_genes * 8 * n_jobs * 1.5) / (1024**3)
            self.logger.info(f"  → Adjusted to n_jobs={n_jobs} (~{est_memory_gb:.1f} GB)")
        
        return n_jobs
    
    def _assess_spatial_quality(self, moran_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess spatial data quality to inform adaptive thresholds.
        
        Returns:
            Dictionary with quality metrics used for adaptive selection
        """
        quality = {
            'moran_df': moran_df,
            'n_genes': len(moran_df),
            'max_moran': moran_df['I'].max(),
            'median_moran': moran_df['I'].median(),
            'mean_moran': moran_df['I'].mean(),
            'q95_moran': moran_df['I'].quantile(0.95),
            'q75_moran': moran_df['I'].quantile(0.75),
            'q50_moran': moran_df['I'].quantile(0.50),
            'q25_moran': moran_df['I'].quantile(0.25),
        }
        
        # Assess if spatial structure is strong/weak/absent
        if quality['median_moran'] > 0.3:
            quality['spatial_strength'] = "strong"
        elif quality['median_moran'] > 0.15:
            quality['spatial_strength'] = "moderate"
        else:
            quality['spatial_strength'] = "weak"
        
        self.logger.info(
            f"  → Spatial quality: {quality['spatial_strength']} "
            f"(median Moran's I = {quality['median_moran']:.3f})"
        )
        
        return quality
    
    def _strict_threshold_selection(self, quality: Dict[str, Any]) -> List[str]:
        """
        Strategy 1: Strict threshold (Human Lymph Node).
        
        Expects discrete compartments with high spatial autocorrelation.
        Uses fixed threshold with safety floor.
        """
        moran_df = quality['moran_df']
        threshold = self.tissue_profile['min_moran_base']
        
        self.logger.info(
            f"  → Applying STRICT threshold: Moran's I > {threshold:.3f}"
        )
        
        # Primary selection: hard threshold
        selected = moran_df[moran_df['I'] > threshold].index.tolist()
        
        # Safety: ensure minimum genes
        if len(selected) < self.target_genes:
            available = len(moran_df)
            requested = min(self.target_genes, available)  # ← Don't request more than available

            self.logger.warning(
                f"  → Only {len(selected)} genes pass threshold. "
                f"Taking top {requested} by Moran's I (max available: {available})."
            )
            selected = moran_df['I'].nlargest(requested).index.tolist()
        
        # Safety: cap at maximum
        if len(selected) > self.max_genes:
            self.logger.info(
                f"  → {len(selected)} genes selected. "
                f"Capping at {self.max_genes}."
            )
            selected = moran_df['I'].nlargest(self.max_genes).index.tolist()
        
        self._log_selection_result(selected, moran_df)
        return selected
    
    def _balanced_threshold_selection(self, quality: Dict[str, Any]) -> List[str]:
        """
        Strategy 2: Balanced threshold (Mouse Brain).
        
        Expects continuous gradients. Uses adaptive threshold that
        adjusts to observed spatial structure strength.
        """
        moran_df = quality['moran_df']
        base_threshold = self.tissue_profile['min_moran_base']
        
        # Adaptive adjustment based on spatial quality
        if quality['spatial_strength'] == "weak":
            # Lower threshold if spatial signal is weak
            adjusted_threshold = base_threshold * 0.7
            self.logger.info(
                f"  → Weak spatial signal detected. "
                f"Lowering threshold to {adjusted_threshold:.3f}"
            )
        elif quality['spatial_strength'] == "strong":
            # Raise threshold if spatial signal is very strong
            adjusted_threshold = base_threshold * 1.2
            self.logger.info(
                f"  → Strong spatial signal detected. "
                f"Raising threshold to {adjusted_threshold:.3f}"
            )
        else:
            adjusted_threshold = base_threshold
        
        self.logger.info(
            f"  → Applying BALANCED threshold: Moran's I > {adjusted_threshold:.3f}"
        )
        
        # Select genes
        selected = moran_df[moran_df['I'] > adjusted_threshold].index.tolist()
        
        # Apply safety bounds
        selected = self._apply_safety_bounds(selected, moran_df)
        
        self._log_selection_result(selected, moran_df)
        return selected
    
    def _robust_percentile_selection(self, quality: Dict[str, Any]) -> List[str]:
        """
        Strategy 3: Robust percentile (Breast Cancer).
        
        Expects heterogeneous/fragmented tissue. Uses percentile-based
        selection that's robust to outliers and fragmentation.
        """
        moran_df = quality['moran_df']
        percentile = self.tissue_profile['moran_percentile']
        
        self.logger.info(
            f"  → Applying ROBUST percentile selection: top {100-percentile}%"
        )
        
        # Calculate percentile threshold
        threshold = moran_df['I'].quantile(percentile / 100)
        
        self.logger.info(
            f"  → Percentile threshold: Moran's I > {threshold:.3f} "
            f"({percentile}th percentile)"
        )
        
        # Select genes above threshold
        selected = moran_df[moran_df['I'] > threshold].index.tolist()
        
        # Additional robustness: if fragmentation is severe, relax further
        if quality['spatial_strength'] == "weak":
            # Take genes with ANY positive spatial autocorrelation
            fallback_threshold = max(0.0, quality['q25_moran'])
            self.logger.warning(
                f"  → Very weak spatial structure. "
                f"Using fallback threshold: {fallback_threshold:.3f}"
            )
            selected = moran_df[moran_df['I'] > fallback_threshold].index.tolist()
        
        # Apply safety bounds
        selected = self._apply_safety_bounds(selected, moran_df)
        
        self._log_selection_result(selected, moran_df)
        return selected
    
    def _apply_safety_bounds(
        self, 
        selected: List[str], 
        moran_df: pd.DataFrame
    ) -> List[str]:
        """
        Ensure selection is within [target_genes, max_genes].
        
        This is critical for downstream stability.
        """
        # Lower bound
        if len(selected) < self.target_genes:
            self.logger.warning(
                f"  → Only {len(selected)} genes selected. "
                f"Expanding to {self.target_genes} (top by Moran's I)."
            )
            selected = moran_df['I'].nlargest(self.target_genes).index.tolist()
        
        # Upper bound
        if len(selected) > self.max_genes:
            self.logger.info(
                f"  → {len(selected)} genes selected. "
                f"Reducing to {self.max_genes} (top by Moran's I)."
            )
            selected = moran_df['I'].nlargest(self.max_genes).index.tolist()
        
        return selected
    
    def _log_selection_result(
        self, 
        selected: List[str], 
        moran_df: pd.DataFrame
    ) -> None:
        """Log final selection statistics."""
        if not selected:
            self.logger.error("No genes selected!")
            return
        
        selected_morans = moran_df.loc[selected, 'I']
        
        self.logger.info(
            f"""
  ✓ SELECTION COMPLETE:
    • Selected: {len(selected)} genes
    • Moran's I range: {selected_morans.min():.3f} to {selected_morans.max():.3f}
    • Moran's I median: {selected_morans.median():.3f}
    • Retention rate: {len(selected) / len(moran_df) * 100:.1f}%
            """
        )
    
    def _fallback_variance_selection(self) -> sc.AnnData:
        """
        Fallback: variance-only selection when spatial methods fail.
        """
        self.logger.warning(
            f"Using variance-only fallback: selecting top {self.target_genes} HVGs"
        )
        
        sc.pp.highly_variable_genes(self.adata, n_top_genes=self.target_genes)
        return self.adata[:, self.adata.var['highly_variable']].copy()

class SpatialPreprocessor:
    """
    Phase 0-1: Preprocessing and Mesh Construction.
    
    Handles:
    1. Gene selection via spatial autocorrelation or variance
    2. Tissue-specific graph construction and pruning
    3. α-complex mesh creation
    4. Heat diffusion smoothing for noise reduction
    
    Advanced Features:
    - Context-Aware Graph Building: Adapts connectivity strategy based on tissue type
    - Vectorized Operations: Eliminates Python loops for graph construction
    - Sparse Matrix Optimization: Memory-efficient heat diffusion for large datasets
    """
    
    def __init__(
        self,
        adata: sc.AnnData,
        config: PERSISTConfig,
        logger: logging.Logger
    ) -> None:
        """
        Initialize the spatial preprocessor.
        
        Args:
            adata: AnnData object with spatial coordinates in .obsm['spatial']
            config: PERSIST configuration parameters
            logger: Logger instance for progress reporting
            
        Raises:
            ValueError: If input data lacks required spatial information
        """
        self.adata: sc.AnnData = adata
        self.config: PERSISTConfig = config
        self.logger: logging.Logger = logger
        self.mesh: Optional[Dict[str, Any]] = None
        self.alpha_complex_wrapper: Optional[Any] = None
        self.simplex_tree: Optional[Any] = None
        self.median_dist: float = 0.0
        
        # Validate input data structure
        self._validate_input_data()
    
    def _validate_input_data(self) -> None:
        """
        Validate input AnnData structure.
        
        Raises:
            ValueError: If required data structures are missing
        """
        if 'spatial' not in self.adata.obsm:
            raise ValueError(
                "AnnData missing .obsm['spatial'] with spatial coordinates. "
                "Please ensure coordinates are stored in adata.obsm['spatial']."
            )
        
        if self.adata.X is None:
            raise ValueError("AnnData has no expression matrix (adata.X is None)")
        
        if self.adata.shape[0] < 10:
            raise ValueError(f"Dataset too small: {self.adata.shape[0]} spots. Need at least 10.")
        
        self.logger.info(f"Input validated: {self.adata.shape[0]} spots, {self.adata.shape[1]} genes")
    
    def run(
        self,
        n_top_genes: int = 2000,
        tissue_type: str = "human_lymph_node"
    ) -> Tuple[sc.AnnData, Optional[Dict[str, Any]], Optional[Any], Optional[Any], float]:
        """
        Execute the preprocessing pipeline with tissue-specific logic.
        
        Args:
            n_top_genes: Number of top genes to select for analysis
            tissue_type: Key matching an entry in TISSUE_SCHEMA (e.g., 'breast_cancer')
            
        Returns:
            Tuple containing:
                - Processed AnnData object
                - Mesh structure dictionary (nodes, edges, triangles)
                - Alpha complex wrapper from Gudhi
                - Simplex tree representation
                - Median nearest-neighbor distance
                
        Raises:
            ValueError: If tissue_type not recognized and no fallback available
            RuntimeError: If mesh construction fails
        """
        self.logger.info("=" * 60)
        self.logger.info("PHASE 0-1: SPATIAL PREPROCESSING AND MESH CONSTRUCTION")
        self.logger.info("=" * 60)
        
        # Step 1: Select informative genes
        self.logger.info("Step 1: Selecting informative genes...")

        # Use the new tissue-adaptive selector
        selector = TissueAdaptiveGeneSelector(
            adata=self.adata,
            logger=self.logger,
            target_genes=n_top_genes,
            max_genes=5000
        )

        self.adata = selector.select_genes(tissue_type=tissue_type)
        
        self.logger.info(f"Selected {self.adata.shape[1]} genes for analysis")
        
        # Step 2: Load Tissue Configuration
        self.logger.info(f"Step 2: Loading configuration for tissue type: {tissue_type}")
        if tissue_type not in TISSUE_SCHEMA:
            self.logger.warning(
                f"Tissue type '{tissue_type}' not found in schema. Using default 'human_lymph_node' settings."
            )
            tissue_type = "human_lymph_node"
        
        # Defaults to 'strict' mode (largest component, no bridging) if tissue is unknown
        default_conf: Dict[str, Any] = {
            "strategy": "largest_component", 
            "min_k": 3, 
            "auto_bridge": False
        }
        tissue_conf: Dict[str, Any] = TISSUE_SCHEMA.get(
            tissue_type, {}
        ).get("mesh_config", default_conf)
        
        self.logger.info(
            f"Mesh strategy: {tissue_conf['strategy']}, "
            f"min_k: {tissue_conf['min_k']}, "
            f"auto_bridge: {tissue_conf.get('auto_bridge', False)}"
        )
        
        # Step 3: Adaptive Mesh Construction
        self.logger.info("Step 3: Constructing adaptive mesh...")
        try:
            self.adata = self._build_mesh_context_aware(tissue_conf)
        except Exception as e:
            self.logger.error(f"Mesh construction failed: {e}")
            raise RuntimeError(f"Mesh construction failed: {e}")
        
        # Step 4: Apply heat diffusion smoothing
        self.logger.info("Step 4: Applying heat diffusion smoothing...")
        self._run_heat_diffusion()
        
        self.logger.info(
            f"Preprocessing complete. "
            f"Final dataset: {self.adata.shape[0]} spots, {self.adata.shape[1]} genes"
        )
        
        return self.adata, self.mesh, self.alpha_complex_wrapper, self.simplex_tree, self.median_dist
    
    def _build_mesh_context_aware(self, conf: Dict[str, Any]) -> sc.AnnData:
        """
        Construct α-complex using context-aware adaptive strategies.
        
        Args:
            conf: Tissue-specific mesh configuration dictionary
            
        Returns:
            AnnData: Processed AnnData with pruned spots
            
        Raises:
            RuntimeError: If mesh construction fails
        """
        coords: np.ndarray = self.adata.obsm["spatial"]
        n_spots: int = coords.shape[0]
        
        self.logger.info(f"Starting with {n_spots} spatial spots")
        
        # 1. Determine connectivity attempts
        min_k: int = conf.get('min_k', 3)
        auto_bridge: bool = conf.get('auto_bridge', False)
        
        # If auto_bridge is True, we try [k, k+2, k+4]. If False, just [k].
        attempts: List[int] = [min_k, min_k + 2, min_k + 4] if auto_bridge else [min_k]
        
        final_mask: Optional[np.ndarray] = None
        
        for k in attempts:
            self.logger.debug(f"Trying connectivity k={k}...")
            
            # --- Vectorized Graph Construction ---
            nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
            indices = nbrs.kneighbors(coords, return_distance=False)
            
            # Create CSR Adjacency Matrix directly from indices
            row = np.repeat(np.arange(n_spots), k)
            col = indices.flatten()
            data = np.ones(n_spots * k, dtype=bool)
            
            adj = sp.csr_matrix((data, (row, col)), shape=(n_spots, n_spots))
            
            # --- Component Analysis ---
            n_comps, labels = connected_components(adj, connection='weak')
            self.logger.debug(f"Found {n_comps} connected components")
            
            # --- Strategy Execution ---
            if conf['strategy'] == 'largest_component':
                # STRICT MODE: Keep only the single largest connected component
                # (Used for Lymph Node, Mouse Brain)
                if n_comps > 1:
                    comp_sizes = np.bincount(labels)
                    giant_idx = np.argmax(comp_sizes)
                    mask = (labels == giant_idx)
                    self.logger.debug(
                        f"Largest component has {comp_sizes[giant_idx]} spots "
                        f"({comp_sizes[giant_idx]/n_spots:.1%} of total)"
                    )
                else:
                    mask = np.ones(n_spots, dtype=bool)
                    
            else:  # 'multi_component'
                # ROBUST MODE: Keep all "biologically significant" islands
                # (Used for Breast Cancer)
                comp_sizes = np.bincount(labels)
                # Threshold: >50 spots OR >1% of total data
                min_size = max(50, int(0.01 * n_spots))
                valid_comps = np.where(comp_sizes >= min_size)[0]
                
                # Vectorized check: which labels are in the valid set?
                mask = np.isin(labels, valid_comps)
                
                self.logger.debug(
                    f"Keeping {len(valid_comps)} components with >{min_size} spots each"
                )
            
            # --- Retention Check ---
            retention: float = np.sum(mask) / n_spots
            
            # If strict mode (no auto-bridge), accept result immediately
            if not auto_bridge:
                final_mask = mask
                self.logger.info(f"Final retention: {retention:.1%} with k={k}")
                break
            
            # If adaptive mode, check if we saved enough data (>85%)
            if retention > 0.85:
                self.logger.info(f"Graph stable at k={k} (Retention: {retention:.1%})")
                final_mask = mask
                break
            else:
                self.logger.warning(
                    f"k={k} resulted in {retention:.1%} retention. "
                    f"Increasing connectivity..."
                )
        
        # If loop finishes without hitting threshold, use the result of the last attempt
        if final_mask is None:
            final_mask = mask
            self.logger.warning(
                f"Max connectivity reached. "
                f"Proceeding with best available graph (retention: {retention:.1%})"
            )
        
        # --- Apply Pruning ---
        n_dropped: int = n_spots - np.sum(final_mask)
        if n_dropped > 0:
            self.logger.info(
                f"Pruning: Dropped {n_dropped} unconnected/noise spots "
                f"({n_dropped/n_spots:.1%} of total)"
            )
            self.adata = self.adata[final_mask].copy()
            coords = self.adata.obsm["spatial"]
        
        # --- Alpha Complex Construction ---
        # Re-compute neighbors on cleaned data for characteristic scale
        nbrs_clean = NearestNeighbors(n_neighbors=2).fit(coords)
        self.median_dist = np.median(nbrs_clean.kneighbors(coords)[0][:, 1])
        self.logger.info(
            f"Characteristic scale (median nearest-neighbor distance): {self.median_dist:.2f}"
        )
        
        # α² = (multiplier * median_dist)²
        master_alpha: float = (self.config.alpha_multiplier * self.median_dist) ** 2
        self.logger.info(f"Maximum alpha squared: {master_alpha:.4f}")
        
        try:
            self.alpha_complex_wrapper = gd.AlphaComplex(points=coords)
            self.simplex_tree = self.alpha_complex_wrapper.create_simplex_tree(
                max_alpha_square=master_alpha
            )
        except Exception as e:
            self.logger.error(f"Alpha complex construction failed: {e}")
            raise RuntimeError(f"Alpha complex construction failed: {e}")
        
        # Extract mesh structure (Vectorized-ish extraction via list comp)
        simplices = list(self.simplex_tree.get_skeleton(2))
        self.mesh = {
            "nodes": np.arange(coords.shape[0]),
            "edges": [tuple(sorted(s[0])) for s in simplices if len(s[0]) == 2],
            "triangles": [tuple(sorted(s[0])) for s in simplices if len(s[0]) == 3]
        }
        
        self.logger.info(
            f"Mesh constructed: {len(self.mesh['nodes'])} nodes, "
            f"{len(self.mesh['edges'])} edges, "
            f"{len(self.mesh['triangles'])} triangles."
        )
        
        # Compute 1D PCA for global signal
        self.logger.debug("Computing 1D PCA for global signal...")
        sc.pp.pca(self.adata, n_comps=1)
        self.adata.obs["X_pca_1d"] = self.adata.obsm['X_pca'][:, 0]
        
        return self.adata
    
    def _run_heat_diffusion(self) -> None:
        """
        Apply heat diffusion smoothing using sparse matrix operations.
        
        Works seamlessly on disconnected graphs (block-diagonal Laplacians).
        Implements normalized graph Laplacian smoothing.
        """
        n: int = len(self.mesh["nodes"])
        
        # 1. Build Adjacency Matrix
        edges = np.array(self.mesh["edges"])
        if len(edges) == 0:
            self.logger.warning("Mesh has no edges. Skipping smoothing.")
            self.adata.obsm["X_smooth"] = self.adata.X
            return
        
        # Symmetrize edges efficiently
        row = np.concatenate([edges[:, 0], edges[:, 1]])
        col = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.ones(len(row))
        
        A = sp.csr_matrix((data, (row, col)), shape=(n, n))
        
        # 2. Compute Normalized Laplacian: L = I - D^-0.5 A D^-0.5
        # Vectorized degree calculation
        deg = np.array(A.sum(1)).flatten()
        
        # Handle isolated nodes (degree = 0)
        isolated_mask = deg == 0
        if np.any(isolated_mask):
            self.logger.warning(f"Found {np.sum(isolated_mask)} isolated nodes in graph")
        
        # Inverse square root of degree (handle 0 safely)
        deg_inv_sqrt = np.power(np.maximum(deg, 1e-10), -0.5)
        # Zero out isolated nodes
        deg_inv_sqrt[isolated_mask] = 0.0
        
        D_inv_sqrt = sp.diags(deg_inv_sqrt)
        
        # 3. Compute Smoothing Operator M
        # M = (I - 0.5 * L) = I - 0.5*(I - S) = 0.5*I + 0.5*S
        # where S = D^-0.5 @ A @ D^-0.5
        try:
            S = D_inv_sqrt @ A @ D_inv_sqrt
            M = 0.5 * sp.eye(n) + 0.5 * S
        except Exception as e:
            self.logger.error(f"Failed to compute smoothing operator: {e}")
            self.adata.obsm["X_smooth"] = self.adata.X
            return
        
        # 4. Apply Diffusion
        X = self.adata.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        
        # Double application for (I - 0.5L)^2
        try:
            self.adata.obsm["X_smooth"] = M @ (M @ X)
            self.logger.info("Heat diffusion smoothing applied successfully")
        except Exception as e:
            self.logger.error(f"Failed to apply heat diffusion: {e}")
            self.adata.obsm["X_smooth"] = self.adata.X


class TopologicalScaleAnalyzer:
    """
    Phase 2: Topological Scale Analysis.
    
    Analyzes persistent homology (H₁) to identify biologically relevant scales.
    The 'birth' times of persistent loops inform which α-complex scales to
    use for subsequent total variation analysis.
    
    Key Insight: Persistent H₁ features correspond to biologically meaningful
    holes/loops in the tissue architecture.
    """
    
    def __init__(
        self,
        simplex_tree: Any,
        median_dist: float,
        alpha_wrapper: Any,
        mesh: Dict[str, Any],
        config: PERSISTConfig,
        logger: logging.Logger
    ) -> None:
        """
        Initialize the topological scale analyzer.
        
        Args:
            simplex_tree: Gudhi simplex tree representation of α-complex
            median_dist: Median nearest-neighbor distance from preprocessing
            alpha_wrapper: Gudhi AlphaComplex wrapper
            mesh: Mesh structure (nodes, edges, triangles)
            config: PERSIST configuration
            logger: Logger instance
        """
        self.simplex_tree: Any = simplex_tree
        self.median_dist: float = median_dist
        self.alpha_wrapper: Any = alpha_wrapper
        self.mesh: Dict[str, Any] = mesh
        self.config: PERSISTConfig = config
        self.logger: logging.Logger = logger
        
        # Results storage
        self.tracked_h1_count: int = 0
        self.persistence_pairs: List[Tuple[float, float]] = []
        self.landscapes: Optional[np.ndarray] = None
    
    def run(self) -> None:
        """
        Compute persistent homology and extract H₁ features.
        
        Performs:
        1. Persistence computation on simplex tree
        2. Extraction of H₁ persistence intervals
        3. Optional persistence landscape computation for visualization
        
        Raises:
            RuntimeError: If persistence computation fails
        """
        self.logger.info("=" * 60)
        self.logger.info("PHASE 2: TOPOLOGICAL SCALE ANALYSIS")
        self.logger.info("=" * 60)
        
        # Compute persistent homology of the α-complex
        self.logger.info("Computing persistent homology...")
        try:
            self.simplex_tree.persistence()
        except Exception as e:
            self.logger.error(f"Persistence computation failed: {e}")
            raise RuntimeError(f"Persistence computation failed: {e}")
        
        # Extract H₁ persistence pairs (birth, death)
        # H₁ corresponds to 1-dimensional holes/loops
        self.persistence_pairs = self.simplex_tree.persistence_intervals_in_dimension(1)
        
        # Count finite persistence pairs (ignore infinite death times)
        finite_pairs = [p for p in self.persistence_pairs if p[1] != float('inf')]
        self.tracked_h1_count = len(finite_pairs)
        
        self.logger.info(
            f"Topological structure: Found {self.tracked_h1_count} "
            f"persistent H₁ features (finite lifetimes)"
        )
        
        # Log persistence statistics if we have features
        if finite_pairs:
            lifetimes = np.array([death - birth for birth, death in finite_pairs])
            self.logger.info(
                f"H₁ persistence statistics: "
                f"mean lifetime = {np.mean(lifetimes):.4f}, "
                f"max lifetime = {np.max(lifetimes):.4f}, "
                f"min lifetime = {np.min(lifetimes):.4f}"
            )
        
        # Compute persistence landscapes if available (for visualization)
        if LANDSCAPE_AVAILABLE and finite_pairs:
            self.logger.debug("Computing persistence landscapes...")
            try:
                self.landscapes = Landscape(
                    num_landscapes=5,
                    resolution=200
                ).fit_transform([np.array(finite_pairs)])
            except Exception as e:
                self.logger.warning(f"Failed to compute persistence landscapes: {e}")
                self.landscapes = None
    
    def compute_optimal_scales(self, n_scales: int) -> np.ndarray:
        """
        Determine optimal α² scales based on H₁ persistence.
        
        Scales are selected to capture biologically relevant topological features:
        - Minimum scale: floor_multiplier * median_dist
        - Maximum scale: 95th percentile of H₁ birth times * 1.5
        - Scales spaced geometrically between min and max
        
        Args:
            n_scales: Desired number of scales
            
        Returns:
            np.ndarray: Array of α² values (squared filtration values)
            
        Raises:
            ValueError: If n_scales is less than 2
        """
        if n_scales < 2:
            raise ValueError(f"n_scales must be at least 2, got {n_scales}")
        
        self.logger.info(f"Computing optimal scales (requested: {n_scales} scales)")
        
        # Minimum scale based on data density
        min_alpha_sq = (self.config.floor_multiplier * self.median_dist) ** 2
        self.logger.debug(f"Minimum alpha squared: {min_alpha_sq:.4f}")
        
        # Extract finite persistence pairs
        finite_pairs = [p for p in self.persistence_pairs if p[1] != float('inf')]
        
        # If no persistent features found, use default range
        if not finite_pairs:
            self.logger.warning(
                "No finite H₁ persistence pairs found. "
                "Using default scale range based on median distance."
            )
            default_max = min_alpha_sq * 10
            scales_sq = np.geomspace(min_alpha_sq, default_max, num=n_scales)
            self.logger.info(
                f"Default scales: {scales_sq[0]:.4f} to {scales_sq[-1]:.4f} "
                f"(geometric spacing)"
            )
            return scales_sq
        
        # Extract birth times (when loops appear)
        birth_sq_values = np.array([p[0] for p in finite_pairs])
        
        # Consider only births above minimum scale
        valid_births = birth_sq_values[birth_sq_values >= min_alpha_sq]
        
        if len(valid_births) == 0:
            self.logger.warning(
                "No valid births above minimum scale. "
                "Using default range."
            )
            default_max = min_alpha_sq * 10
            scales_sq = np.geomspace(min_alpha_sq, default_max, num=n_scales)
            return scales_sq
        
        # Use 95th percentile of valid births as upper bound
        max_birth_sq = np.percentile(valid_births, 95)
        
        # Generate scales geometrically between min and max
        scales_sq = np.geomspace(
            min_alpha_sq,
            max_birth_sq * 1.5,  # Add 50% buffer
            num=n_scales
        )
        
        # Ensure scales are sorted
        scales_sq = np.sort(scales_sq)
        
        self.logger.info(
            f"Optimal scales computed: {scales_sq[0]:.4f} to {scales_sq[-1]:.4f} "
            f"({n_scales} scales, geometric spacing)"
        )
        self.logger.debug(f"Scale values: {np.round(scales_sq, 4)}")
        
        return scales_sq
    
    def compute_scale_weights(self, query_scales_sq: np.ndarray) -> np.ndarray:
        """
        Compute weights for each scale based on total persistence of active H₁ features.
        
        Scales with more persistent topological activity receive higher weights.
        This emphasizes biologically meaningful scales in the multiscale analysis.
        
        Args:
            query_scales_sq: Array of α² scales to weight
            
        Returns:
            np.ndarray: Array of weights normalized to mean 1.0
            
        Raises:
            ValueError: If query_scales_sq is empty
        """
        if len(query_scales_sq) == 0:
            raise ValueError("query_scales_sq cannot be empty")
        
        self.logger.debug("Computing scale weights based on H₁ persistence...")
        
        weights = []
        finite_pairs = [p for p in self.persistence_pairs if p[1] != float('inf')]
        
        for i, alpha_sq in enumerate(query_scales_sq):
            # Identify H₁ features active at this scale
            # (born before scale, die after scale)
            active_pairs = [
                p for p in finite_pairs
                if p[0] <= alpha_sq <= p[1]
            ]
            
            if not active_pairs:
                # Minimum weight for scales without active features
                weights.append(0.1)
            else:
                # Weight proportional to total persistence of active features
                total_persistence = sum(p[1] - p[0] for p in active_pairs)
                weights.append(total_persistence + 0.1)  # Add small constant
        
        weights = np.array(weights)
        
        # Normalize weights to have mean 1.0
        if np.mean(weights) > 0:
            weights = weights / np.mean(weights)
        else:
            weights = np.ones_like(weights)  # Uniform weights if all zero
        
        self.logger.debug(f"Scale weights (normalized): {np.round(weights, 2)}")
        
        return weights


class MultiscaleTVOperator:
    """
    Phase 3: Multiscale Total Variation Analysis.
    
    Updated: Includes 'Frontier Masking' to prevent edge artifacts
    in fragmented tissues (like Breast Cancer).
    
    Computes total variation across multiple scales of the α-complex,
    integrating topological information from persistence analysis.
    """
    
    def __init__(
        self,
        adata: sc.AnnData,
        mesh: Dict[str, Any],
        alpha_wrapper: Any,
        config: PERSISTConfig,
        logger: logging.Logger
    ) -> None:
        """
        Initialize the multiscale TV operator.
        
        Args:
            adata: AnnData object with processed gene expression
            mesh: Mesh structure from preprocessing
            alpha_wrapper: Gudhi AlphaComplex wrapper
            config: PERSIST configuration
            logger: Logger instance
        """
        self.adata: sc.AnnData = adata
        self.mesh: Dict[str, Any] = mesh
        self.alpha_wrapper: Any = alpha_wrapper
        self.config: PERSISTConfig = config
        self.logger: logging.Logger = logger
        
        # Store TV stability at each scale for downstream analysis
        self.scale_history: Dict[float, np.ndarray] = {}
        
        # Pre-calculate the boundary mask immediately
        self.boundary_mask: np.ndarray = self._identify_topological_boundaries()
    
    def _identify_topological_boundaries(self) -> np.ndarray:
        """
        Mathematically identifies the 'skin' of the tissue mesh.
        
        Used to suppress TV artifacts at the physical edge of the tissue.
        Boundary edges are those that belong to only one triangle.
        
        Returns:
            np.ndarray: Boolean mask where True indicates boundary nodes
        """
        self.logger.debug("Identifying topological boundaries for frontier masking...")
        
        # Vectorized edge counting
        tris = np.array(self.mesh["triangles"])
        if len(tris) == 0:
            self.logger.warning("Mesh has no triangles. Boundary mask will be empty.")
            return np.zeros(len(self.mesh["nodes"]), dtype=bool)
        
        # Break triangles into 3 edges each
        edges = np.concatenate([
            tris[:, [0, 1]],
            tris[:, [1, 2]],
            tris[:, [0, 2]]
        ])
        edges.sort(axis=1)
        
        # Fast unique counting using void view (much faster than loops)
        edges_view = np.ascontiguousarray(edges).view(
            np.dtype((np.void, edges.dtype.itemsize * 2))
        )
        _, idx, counts = np.unique(
            edges_view, return_index=True, return_counts=True
        )
        
        # Boundary edges appear exactly once (shared edges appear twice)
        boundary_edge_indices = idx[counts == 1]
        boundary_edges = edges[boundary_edge_indices]
        
        # Identify all nodes involved in boundary edges
        boundary_nodes = np.unique(boundary_edges.flatten())
        
        n_nodes = len(self.mesh["nodes"])
        mask = np.zeros(n_nodes, dtype=bool)
        mask[boundary_nodes] = True
        
        self.logger.info(
            f"Topological Boundary Mask: identified {len(boundary_nodes)} "
            f"edge nodes to suppress ({len(boundary_nodes)/n_nodes:.1%} of total)"
        )
        
        return mask
    
    def run(
        self,
        alpha_scales_sq: np.ndarray,
        topology_analyzer: TopologicalScaleAnalyzer
    ) -> None:
        """
        Execute multiscale total variation analysis.
        
        Args:
            alpha_scales_sq: Array of α² scales to analyze
            topology_analyzer: TopologicalScaleAnalyzer instance for scale weights
            
        Raises:
            ValueError: If alpha_scales_sq is empty or invalid
            RuntimeError: If TV computation fails
        """
        self.logger.info("=" * 60)
        self.logger.info("PHASE 3: MULTISCALE TOTAL VARIATION ANALYSIS")
        self.logger.info("=" * 60)
        
        if len(alpha_scales_sq) == 0:
            raise ValueError("alpha_scales_sq cannot be empty")
        
        # Ensure 1D PCA is computed
        if "X_pca_1d" not in self.adata.obs:
            self.logger.info("Computing 1D PCA for global signal...")
            try:
                sc.pp.pca(self.adata, n_comps=1)
                self.adata.obs["X_pca_1d"] = self.adata.obsm['X_pca'][:, 0]
            except Exception as e:
                self.logger.error(f"Failed to compute PCA: {e}")
                raise RuntimeError(f"PCA computation failed: {e}")
        
        signal = self.adata.obs["X_pca_1d"].values
        signal_filled = np.nan_to_num(signal, nan=0.0)
        
        if GPU_AVAILABLE:
            signal_gpu = cp.array(signal_filled)
            self.logger.info("Using GPU acceleration for TV computations")
        else:
            self.logger.info("Using CPU for TV computations")
        
        # Pre-compute complex for all scales
        max_scale = alpha_scales_sq.max() * 1.1  # 10% buffer
        try:
            st = self.alpha_wrapper.create_simplex_tree(max_alpha_square=max_scale)
            skel = list(st.get_skeleton(2))
            edge_list = [tuple(sorted(s[0])) for s in skel if len(s[0]) == 2]
            edge_alphas_sq = np.array([s[1] for s in skel if len(s[0]) == 2])
            edges_np = np.array(edge_list)
        except Exception as e:
            self.logger.error(f"Failed to create simplex tree: {e}")
            raise RuntimeError(f"Simplex tree creation failed: {e}")
        
        # Get scale weights from topology analyzer
        weights = topology_analyzer.compute_scale_weights(alpha_scales_sq)
        self.logger.info(f"Scale weights assigned: {np.round(weights, 2)}")
        
        # Initialize accumulation arrays
        accumulated_stability = np.zeros(self.adata.shape[0])
        total_weight = 0.0
        
        # Progress bar for scale iteration
        progress_bar = tqdm(
            enumerate(alpha_scales_sq),
            total=len(alpha_scales_sq),
            desc="Computing TV across scales",
            file=sys.stdout,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} scales'
        )
        
        for i, alpha_sq in progress_bar:
            self.logger.debug(f"Processing scale {i+1}/{len(alpha_scales_sq)}: α²={alpha_sq:.4f}")
            
            # Build boundary operator for current scale
            B = self._build_boundary_operator_consistent(alpha_sq, edges_np, edge_alphas_sq)
            
            # Compute total variation
            if GPU_AVAILABLE:
                B_gpu = cpsp.csr_matrix(B)
                edge_diffs = cp.abs(B_gpu.T @ signal_gpu)
                node_tv = cp.asnumpy(cp.abs(B_gpu) @ edge_diffs)
            else:
                edge_diffs = np.abs(B.T @ signal)
                node_tv = np.abs(B) @ edge_diffs
            
            # Normalize by vertex degree if configured
            if self.config.degree_normalize_tv:
                degree = np.array(np.abs(B).sum(axis=1)).flatten()
                degree[degree == 0] = 1.0  # Avoid division by zero
                node_tv = node_tv / degree
            
            # Apply Frontier Masking: suppress TV at the outer boundary
            node_tv[self.boundary_mask] = np.median(node_tv)
            
            # Clip extreme values and compute stability
            p05, p95 = self.config.grad_clip_percentiles
            tv_clipped = np.clip(
                node_tv,
                np.percentile(node_tv, p05),
                np.percentile(node_tv, p95)
            )
            
            # Normalize to [0, 1] range for stability score
            denom = (np.percentile(node_tv, p95) - np.percentile(node_tv, p05) + 1e-9)
            stability = 1.0 - ((tv_clipped - np.percentile(node_tv, p05)) / denom)
            
            # Accumulate weighted stability
            w = weights[i]
            accumulated_stability += stability * w
            total_weight += w
            
            # Store scale history
            self.scale_history[alpha_sq] = stability
            
            # Update progress bar
            progress_bar.set_postfix({
                'weight': f'{w:.2f}',
                'stability_mean': f'{np.mean(stability):.3f}'
            })
        
        # Compute final stability score
        self.adata.obs["stability_score"] = accumulated_stability / (total_weight + 1e-9)
        self.logger.info(
            f"Stability score computed: "
            f"mean={np.mean(self.adata.obs['stability_score']):.3f}, "
            f"std={np.std(self.adata.obs['stability_score']):.3f}"
        )
        
        # Compute frontier TV (Walls) at minimum scale
        self._compute_frontier_tv(
            alpha_scales_sq[0], signal, edges_np, edge_alphas_sq
        )
        
        self.logger.info("Multiscale TV analysis completed successfully")
    
    def _compute_frontier_tv(
        self,
        min_scale_sq: float,
        signal: np.ndarray,
        edges_np: np.ndarray,
        edge_alphas_sq: np.ndarray
    ) -> None:
        """
        Compute the 'Wall' signal with boundary suppression.
        
        The frontier TV highlights strong transcriptional boundaries
        while suppressing edge artifacts.
        
        Args:
            min_scale_sq: Minimum α² scale
            signal: 1D PCA signal
            edges_np: Array of mesh edges
            edge_alphas_sq: Array of edge filtration values
        """
        self.logger.debug("Computing frontier TV (walls)...")
        
        # Build boundary operator at minimum scale
        B = self._build_boundary_operator_consistent(
            min_scale_sq, edges_np, edge_alphas_sq
        )
        
        # Compute TV magnitude
        if GPU_AVAILABLE:
            B_gpu = cpsp.csr_matrix(B)
            edge_diffs = cp.abs(B_gpu.T @ cp.array(signal))
            tv = cp.asnumpy(cp.abs(B_gpu) @ edge_diffs)
        else:
            edge_diffs = np.abs(B.T @ signal)
            tv = np.abs(B) @ edge_diffs
        
        # Normalize by degree if configured
        if self.config.degree_normalize_tv:
            degree = np.array(np.abs(B).sum(axis=1)).flatten()
            degree[degree == 0] = 1.0
            tv = tv / degree
        
        # Apply boundary mask to walls (Force walls at edges to zero)
        tv[self.boundary_mask] = 0.0
        
        # Clip and normalize to [0, 1]
        tv_magnitude = np.clip(
            tv / np.percentile(tv, self.config.frontier_clip_percentile),
            0, 1
        )
        
        self.adata.obs["tv_magnitude"] = tv_magnitude
        self.logger.info(
            f"Frontier TV computed: "
            f"mean magnitude={np.mean(tv_magnitude):.3f}, "
            f"max magnitude={np.max(tv_magnitude):.3f}"
        )
    
    def compute_robustness(self) -> None:
        """
        Assess robustness of stability cores via bootstrapping.
        
        Performs multiple bootstrap resampling iterations to identify
        consistently stable regions across subsamples.
        """
        self.logger.info("Phase 3b: Bootstrapping for robustness assessment")
        
        stab = self.adata.obs["stability_score"].values
        coords = self.adata.obsm["spatial"]
        n_spots = len(coords)
        
        robustness_count = np.zeros(n_spots)
        
        self.logger.info(
            f"Running {self.config.bootstrap_runs} bootstrap iterations "
            f"(sample fraction: {self.config.bootstrap_sample_frac})"
        )
        
        for i in range(self.config.bootstrap_runs):
            self.logger.debug(f"Bootstrap iteration {i+1}/{self.config.bootstrap_runs}")
            
            # Resample without replacement
            indices = resample(
                np.arange(n_spots),
                n_samples=int(self.config.bootstrap_sample_frac * n_spots),
                replace=False
            )
            
            # Map stability scores back to original coordinates
            grid_stab = griddata(
                coords[indices], stab[indices],
                (coords[:, 0], coords[:, 1]),
                method='nearest'
            )
            
            # Count spots in top 35% of stability
            threshold = np.nanpercentile(grid_stab, 65)
            robustness_count += (grid_stab > threshold)
        
        # Compute robustness score
        self.adata.obs["tv_robustness"] = robustness_count / self.config.bootstrap_runs
        
        # Identify significant cores
        self.adata.obs["is_significant_core"] = (
            self.adata.obs["tv_robustness"] > self.config.bootstrap_significance
        )
        
        n_significant = np.sum(self.adata.obs["is_significant_core"])
        self.logger.info(
            f"Robustness assessment complete: "
            f"{n_significant} significant cores identified "
            f"({n_significant/n_spots:.1%} of spots)"
        )
    
    def _build_boundary_operator_consistent(
        self,
        alpha_sq: float,
        full_edges: np.ndarray,
        full_alpha_sq: np.ndarray
    ) -> sp.csr_matrix:
        """
        Build boundary operator for a specific α² scale.
        
        Args:
            alpha_sq: Current α² scale
            full_edges: All edges in the complete complex
            full_alpha_sq: Filtration values for all edges
            
        Returns:
            scipy.sparse.csr_matrix: Boundary operator matrix
        """
        # Filter edges active at current scale
        mask = full_alpha_sq <= alpha_sq
        active_edges = full_edges[mask]
        n_vertices = len(self.mesh['nodes'])
        n_edges = len(active_edges)
        
        if n_edges == 0:
            # Return empty matrix if no edges active
            return sp.csr_matrix((n_vertices, 0))
        
        # Construct boundary operator: incidence matrix with ±1 entries
        v0 = active_edges[:, 0]
        v1 = active_edges[:, 1]
        row_indices = np.concatenate([v0, v1])
        col_indices = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
        data = np.concatenate([-np.ones(n_edges), np.ones(n_edges)])
        
        return sp.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_vertices, n_edges)
        )


class TopographicBasinDecomposer:
    """
    Phase 4: Basin Decomposition via Watershed Segmentation.
    
    Decomposes the stability scalar field into basins of attraction
    using watershed segmentation on a smoothed grid interpolation.
    
    Key Insight: Stability basins correspond to tissue microdomains
    with coherent transcriptional programs.
    """
    
    def __init__(
        self,
        adata: sc.AnnData,
        median_dist: float,
        mesh: Dict[str, Any],
        config: PERSISTConfig,
        logger: logging.Logger
    ) -> None:
        """
        Initialize the basin decomposer.
        
        Args:
            adata: AnnData with stability scores
            median_dist: Median nearest-neighbor distance for grid resolution
            mesh: Mesh structure
            config: PERSIST configuration
            logger: Logger instance
        """
        self.adata: sc.AnnData = adata
        self.median_dist: float = median_dist
        self.mesh: Dict[str, Any] = mesh
        self.config: PERSISTConfig = config
        self.logger: logging.Logger = logger
    
    def run(self) -> None:
        """
        Perform watershed segmentation on stability field.
        
        Steps:
        1. Interpolate stability scores onto regular grid
        2. Apply Gaussian smoothing for noise reduction
        3. Detect local maxima as watershed seeds
        4. Apply watershed segmentation
        5. Map labels back to original coordinates
        
        Raises:
            RuntimeError: If watershed segmentation fails
        """
        self.logger.info("=" * 60)
        self.logger.info("PHASE 4: BASIN DECOMPOSITION (TOPOGRAPHIC WATERSHED)")
        self.logger.info("=" * 60)
        
        # Extract stability scores and coordinates
        stab = self.adata.obs["tv_robustness"].values
        coords = self.adata.obsm["spatial"]
        n_spots = len(coords)
        
        self.logger.info(f"Performing watershed on {n_spots} spots")
        
        # Determine grid resolution based on spatial density
        # Aim for 2x oversampling relative to median distance
        raw_res_x = int(
            (coords[:, 0].max() - coords[:, 0].min()) / (self.median_dist * 0.5)
        )
        raw_res_y = int(
            (coords[:, 1].max() - coords[:, 1].min()) / (self.median_dist * 0.5)
        )
        
        # Cap resolution for memory efficiency
        res_x = min(raw_res_x, self.config.max_grid_dim)
        res_y = min(raw_res_y, self.config.max_grid_dim)
        
        if res_x < raw_res_x or res_y < raw_res_y:
            self.logger.info(
                f"Grid resolution capped to {res_x}x{res_y} "
                f"(original: {raw_res_x}x{raw_res_y})"
            )
        
        # Create interpolation grid
        grid_x, grid_y = np.mgrid[
            coords[:, 0].min():coords[:, 0].max():complex(res_x),
            coords[:, 1].min():coords[:, 1].max():complex(res_y)
        ]
        
        # Interpolate stability scores onto grid
        self.logger.debug("Interpolating stability scores to grid...")
        grid_z = griddata(
            coords, stab,
            (grid_x, grid_y),
            method='linear',
            fill_value=0
        )
        
        # Apply Gaussian smoothing for noise reduction
        self.logger.debug(
            f"Applying Gaussian smoothing (sigma={self.config.watershed_sigma})..."
        )
        grid_z = gaussian_filter(grid_z, sigma=self.config.watershed_sigma)
        
        # Detect local maxima as watershed seeds
        self.logger.debug("Detecting local maxima for watershed seeds...")
        markers = np.zeros_like(grid_z, dtype=int)
        peaks = peak_local_max(
            grid_z,
            min_distance=10,
            threshold_abs=self.config.watershed_threshold_abs
        )
        
        for i, peak in enumerate(peaks):
            markers[peak[0], peak[1]] = i + 1
        
        self.logger.info(f"Found {len(peaks)} local maxima for watershed seeds")
        
        # Apply watershed segmentation
        # Use negative stability so basins correspond to high stability
        self.logger.debug("Applying watershed segmentation...")
        try:
            labels_grid = watershed(
                -grid_z,  # Invert for watershed (basins = high stability)
                markers,
                mask=grid_z > self.config.watershed_mask_threshold
            )
        except Exception as e:
            self.logger.error(f"Watershed segmentation failed: {e}")
            raise RuntimeError(f"Watershed segmentation failed: {e}")
        
        # Count unique basins (excluding background label 0)
        n_basins = len(np.unique(labels_grid)) - 1
        self.logger.info(f"Watershed segmentation identified {n_basins} stability basins")
        
        # Map grid labels back to original coordinates
        self.logger.debug("Mapping watershed labels back to original coordinates...")
        valid = labels_grid.ravel() > 0
        interpolator = NearestNDInterpolator(
            np.c_[grid_x.ravel()[valid], grid_y.ravel()[valid]],
            labels_grid.ravel()[valid]
        )
        
        # Assign domain labels to each spot
        domain_labels = interpolator(coords).astype(int).astype(str)
        self.adata.obs["stability_domain"] = pd.Categorical(domain_labels)
        
        # Report domain statistics
        domain_counts = self.adata.obs["stability_domain"].value_counts()
        self.logger.info(
            f"Basin decomposition complete. Domain sizes: "
            f"min={domain_counts.min()}, max={domain_counts.max()}, "
            f"mean={domain_counts.mean():.1f}"
        )


class PERSISTMetrics:
    """
    Phase 5a: Quantitative Analysis & Annotation.
    
    Responsibility:
    - Calculates validation metrics (ARI, TOI, SLC, Correlation).
    - Aligns stability domains with biological markers.
    - Saves statistical results to JSON.
    """
    
    def __init__(
        self,
        adata: sc.AnnData,
        tv_operator: MultiscaleTVOperator,
        config: PERSISTConfig,
        output_dir: str,
        logger: logging.Logger
    ) -> None:
        """
        Initialize metrics calculator.
        
        Args:
            adata: AnnData with computed stability domains
            tv_operator: MultiscaleTVOperator instance
            config: PERSIST configuration
            output_dir: Directory for saving metrics
            logger: Logger instance
        """
        self.adata: sc.AnnData = adata
        self.tv: MultiscaleTVOperator = tv_operator
        self.config: PERSISTConfig = config
        self.output_dir: str = output_dir
        self.logger: logging.Logger = logger
        self.stats: Dict[str, Any] = {}
    
    def run(
        self,
        reference_markers: Optional[List[str]] = None,
        stability_markers: Optional[Dict[str, List[str]]] = None,
        baseline_key: str = 'leiden'
    ) -> Dict[str, Any]:
        """
        Execute all metric calculations.
        
        Args:
            reference_markers: List of reference genes for validation
            stability_markers: Dictionary mapping cell types to marker genes
            baseline_key: Key in adata.obs for baseline clustering comparison
            
        Returns:
            Dict[str, Any]: Dictionary of computed metrics
        """
        self.logger.info("=" * 60)
        self.logger.info("PHASE 5a: QUANTITATIVE METRICS AND ANNOTATION")
        self.logger.info("=" * 60)
        
        # 1. Biological Alignment (Annotation)
        self.logger.info("Step 1: Biological annotation alignment...")
        self._perform_external_annotation_alignment(stability_markers)
        
        # 2. Structural Metrics
        self.logger.info("Step 2: Structural metric computation...")
        self._benchmark_clustering_baseline(baseline_key)
        self._validate_orthogonality(baseline_key)
        self._validate_scale_coherence()
        
        # 3. Coupling Metrics
        self.logger.info("Step 3: Coupling metric computation...")
        if reference_markers:
            self._validate_gradient_reference_coupling(reference_markers)
        self._validate_stability_function_coupling(stability_markers)
        
        # 4. Save results
        self.logger.info("Step 4: Saving metrics...")
        self._save_metrics()
        
        # Log summary
        self._log_metric_summary()
        
        return self.stats
    
    def _save_metrics(self) -> None:
        """Save computed stats to JSON."""
        metrics_file = os.path.join(self.output_dir, "persist_metrics.json")
        
        # Ensure data is JSON serializable
        clean_stats = {}
        for key, value in self.stats.items():
            if isinstance(value, (float, int, np.number)):
                clean_stats[key] = float(value)
            elif isinstance(value, dict):
                clean_stats[key] = {k: float(v) for k, v in value.items()}
            elif isinstance(value, list):
                clean_stats[key] = [float(v) if isinstance(v, (float, int, np.number)) else v 
                                   for v in value]
            else:
                clean_stats[key] = str(value)
        
        try:
            with open(metrics_file, "w") as f:
                json.dump(clean_stats, f, indent=4, sort_keys=True)
            self.logger.info(f"Metrics saved to {metrics_file}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    def _log_metric_summary(self) -> None:
        """Log a summary of key metrics."""
        summary_lines = ["Metric Summary:"]
        
        if 'Baseline_ARI' in self.stats:
            summary_lines.append(
                f"  Baseline ARI: {self.stats['Baseline_ARI']:.4f}"
            )
        if 'TOI' in self.stats:
            summary_lines.append(f"  Topological Orthogonality Index: {self.stats['TOI']:.4f}")
        if 'SLC' in self.stats:
            summary_lines.append(f"  Scale Local Coherence: {self.stats['SLC']:.4f}")
        if 'Gradient_Reference_Coupling_Rho' in self.stats:
            summary_lines.append(
                f"  Gradient-Reference Coupling: {self.stats['Gradient_Reference_Coupling_Rho']:.4f}"
            )
        
        self.logger.info("\n".join(summary_lines))
    
    def _get_valid_markers(self, candidates: List[str]) -> List[str]:
        """
        Filter markers to those present in the dataset.
        
        Args:
            candidates: List of candidate marker genes
            
        Returns:
            List[str]: Valid markers present in dataset
        """
        if not candidates:
            return []
        
        available = set(self.adata.var_names)
        valid = []
        
        for gene in candidates:
            # Check for case sensitivity variations
            if gene in available:
                valid.append(gene)
            elif gene.upper() in available:
                valid.append(gene.upper())
            elif gene.title() in available:
                valid.append(gene.title())
            else:
                self.logger.debug(f"Marker gene not found: {gene}")
        
        return valid
    
    def _perform_external_annotation_alignment(
        self,
        stability_markers: Optional[Dict[str, List[str]]]
    ) -> None:
        """
        Align stability domains with biological annotations.
        
        Args:
            stability_markers: Dictionary mapping cell types to marker genes
        """
        markers = stability_markers if stability_markers else {}
        
        # Filter valid markers
        available = {
            k: self._get_valid_markers(v) for k, v in markers.items()
        }
        available = {k: v for k, v in available.items() if v}
        
        # If no markers available, just copy stability domain
        if not available:
            self.logger.warning(
                "No valid stability markers found. "
                "Using stability domain labels directly."
            )
            self.adata.obs['domain_annotation'] = self.adata.obs['stability_domain']
            return
        
        self.logger.info(
            f"Aligning domains with {len(available)} biological annotations"
        )
        
        # Calculate global background for normalization
        global_means = {}
        for key, m_list in available.items():
            try:
                global_means[key] = np.array(
                    self.adata[:, m_list].X.mean()
                ).item() + 1e-9
            except Exception as e:
                self.logger.warning(f"Failed to compute mean for {key}: {e}")
                global_means[key] = 1.0
        
        # Map each domain to best fitting annotation
        mapping = {}
        domain_categories = self.adata.obs['stability_domain'].cat.categories
        
        for domain in domain_categories:
            mask = self.adata.obs['stability_domain'] == domain
            if mask.sum() == 0:
                continue
            
            scores = {}
            for key, m_list in available.items():
                try:
                    d_expr = np.array(
                        self.adata[mask, m_list].X.mean()
                    ).item()
                    scores[key] = d_expr / global_means[key]
                except Exception as e:
                    self.logger.debug(f"Failed to score {key} for domain {domain}: {e}")
                    scores[key] = 0.0
            
            # Require >0.5 fold enrichment over mean to annotate
            if scores and max(scores.values()) > 0.5:
                best = max(scores, key=scores.get)
                # Shorten annotation: "B_Cell (Core)" -> "B_Cell"
                mapping[domain] = f"{domain}: {best.split(' ')[0]}"
                self.logger.debug(
                    f"Domain {domain} annotated as {best} "
                    f"(enrichment: {scores[best]:.2f})"
                )
            else:
                mapping[domain] = f"{domain}"
                self.logger.debug(
                    f"Domain {domain} has no clear annotation "
                    f"(max enrichment: {max(scores.values()) if scores else 0:.2f})"
                )
        
        self.adata.obs['domain_annotation'] = (
            self.adata.obs['stability_domain'].map(mapping)
        )
        
        # Report annotation statistics
        unique_annotations = self.adata.obs['domain_annotation'].unique()
        self.logger.info(
            f"Annotation complete: {len(unique_annotations)} unique annotations"
        )
    
    def _benchmark_clustering_baseline(self, baseline_key: str) -> None:
        """
        Compute ARI against standard clustering.
        
        Args:
            baseline_key: Key in adata.obs for baseline clustering
        """
        # Compute baseline clustering if not present
        if baseline_key not in self.adata.obs:
            self.logger.info(f"Computing baseline {baseline_key} clustering...")
            try:
                sc.pp.neighbors(self.adata, use_rep='X_smooth')
                sc.tl.leiden(self.adata, resolution=0.5, key_added=baseline_key)
                self.logger.info(
                    f"Baseline clustering created: "
                    f"{self.adata.obs[baseline_key].nunique()} clusters"
                )
            except Exception as e:
                self.logger.error(f"Failed to compute baseline clustering: {e}")
                return
        
        # Compute Adjusted Rand Index
        try:
            ari = adjusted_rand_score(
                self.adata.obs['stability_domain'].cat.codes,
                self.adata.obs[baseline_key].cat.codes
            )
            self.stats['Baseline_ARI'] = ari
            self.logger.info(f"Baseline ARI: {ari:.4f}")
        except Exception as e:
            self.logger.error(f"Failed to compute ARI: {e}")
            self.stats['Baseline_ARI'] = 0.0
    
    def _validate_orthogonality(self, baseline_key: str) -> None:
        """
        Calculate Topological Orthogonality Index (TOI).
        
        TOI measures independence between stability scores and clustering,
        where 1.0 indicates complete orthogonality.
        
        Args:
            baseline_key: Key in adata.obs for baseline clustering
        """
        try:
            stab = self.adata.obs['stability_score'].values
            clusters = pd.get_dummies(self.adata.obs[baseline_key])
            
            max_corr = 0.0
            for col in clusters.columns:
                corr = abs(np.corrcoef(stab, clusters[col].values)[0, 1])
                max_corr = max(max_corr, corr)
            
            self.stats['TOI'] = 1.0 - max_corr
            self.logger.info(f"Topological Orthogonality Index: {self.stats['TOI']:.4f}")
        except Exception as e:
            self.logger.error(f"Failed to compute TOI: {e}")
            self.stats['TOI'] = 0.0
    
    def _validate_scale_coherence(self) -> None:
        """Calculate Scale Local Coherence (SLC)."""
        if len(self.tv.scale_history) < 2:
            self.logger.warning(
                "Insufficient scales for SLC calculation. "
                "Need at least 2 scales in scale_history."
            )
            self.stats['SLC'] = 0.0
            return
        
        try:
            energies = list(self.tv.scale_history.values())
            jaccards = []
            
            for i in range(len(energies) - 1):
                # Compare top 30% most stable regions between adjacent scales
                m1 = energies[i] > np.percentile(energies[i], 70)
                m2 = energies[i + 1] > np.percentile(energies[i + 1], 70)
                
                intersection = np.sum(m1 & m2)
                union = np.sum(m1 | m2)
                jaccards.append(intersection / (union + 1e-9))
            
            self.stats['SLC'] = np.mean(jaccards)
            self.logger.info(f"Scale Local Coherence: {self.stats['SLC']:.4f}")
        except Exception as e:
            self.logger.error(f"Failed to compute SLC: {e}")
            self.stats['SLC'] = 0.0
    
    def _validate_gradient_reference_coupling(self, features: List[str]) -> None:
        """
        Correlate TV Walls with biological boundary markers.
        
        Args:
            features: List of reference marker genes
        """
        tv_field = np.nan_to_num(self.adata.obs["tv_magnitude"].values)
        valid = self._get_valid_markers(features)
        
        if not valid:
            self.logger.warning("No valid reference markers found for gradient coupling.")
            return
        
        try:
            ref_signal = np.nan_to_num(
                np.array(self.adata[:, valid].X.mean(axis=1)).flatten()
            )
            corr, p = spearmanr(tv_field, ref_signal)
            
            self.stats['Gradient_Reference_Coupling_Rho'] = corr
            self.stats['Gradient_Reference_Coupling_Pval'] = p
            self.logger.info(
                f"Gradient-Reference Coupling: rho={corr:.4f}, p={p:.4e}"
            )
        except Exception as e:
            self.logger.error(f"Failed to compute gradient-reference coupling: {e}")
            self.stats['Gradient_Reference_Coupling_Rho'] = 0.0
            self.stats['Gradient_Reference_Coupling_Pval'] = 1.0
    
    def _validate_stability_function_coupling(
        self,
        markers: Optional[Dict[str, List[str]]]
    ) -> None:
        """
        Correlate stability basins with functional markers.
        
        Args:
            markers: Dictionary mapping cell types to marker genes
        """
        if not markers:
            return
        
        try:
            stab = np.nan_to_num(self.adata.obs["stability_score"].values)
            corrs = {}
            
            for name, genes in markers.items():
                valid = self._get_valid_markers(genes)
                if not valid:
                    continue
                
                sig = np.nan_to_num(
                    np.array(self.adata[:, valid].X.mean(axis=1)).flatten()
                )
                corr = np.corrcoef(stab, sig)[0, 1]
                corrs[name] = corr
            
            self.stats['Stability_Correlations'] = corrs
            
            # Log top correlations
            if corrs:
                top_corr = max(corrs.items(), key=lambda x: abs(x[1]))
                self.logger.info(
                    f"Strongest stability correlation: "
                    f"{top_corr[0]} = {top_corr[1]:.4f}"
                )
        except Exception as e:
            self.logger.error(f"Failed to compute stability-function coupling: {e}")
            self.stats['Stability_Correlations'] = {}


class PERSISTVisualizer:
    """
    Phase 5b: Professional Visualization Suite (PDF Report + Individual PNGs).
    
    Generates a comprehensive PDF booklet AND individual high-res images.
    
    Updates:
    - Saves individual PNGs to 'visualizations/' folder.
    - Added Overlay plot (Domains on Histology).
    - Solid (opaque) plots for maps; Transparent for overlays.
    - Descriptive legends (e.g., "0=Unstable, 1=Stable").
    - Refined Saliency plot logic.
    """
    
    def __init__(
        self,
        adata: sc.AnnData,
        mesh: Dict[str, Any],
        topo: Any,
        tv_operator: Any,
        config: Any,
        output_dir: str,
        logger: logging.Logger
    ) -> None:
        self.adata = adata
        self.mesh = mesh
        self.topo = topo
        self.tv = tv_operator
        self.config = config
        self.output_dir = output_dir
        self.logger = logger
        
        # Ensure directory exists
        self.viz_dir = os.path.join(output_dir, "visualizations")
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)
            
        self.saliency_df = None

    def run(self, stats: Dict[str, Any], reference_markers=None, stability_markers=None) -> None:
        """Execute visualization pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 5b: GENERATING VISUALIZATIONS & REPORT")
        self.logger.info("=" * 60)

        # 1. Compute metrics
        self.saliency_df = self._compute_domain_saliency()
        
        # 2. Generate PDF and PNGs
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_path = os.path.join(self.output_dir, "PERSIST_Analysis_Report.pdf")
        
        try:
            with PdfPages(pdf_path) as pdf:
                # --- Overview ---
                self._page_title_summary(pdf, stats)
                self._page_histology(pdf)
                
                # --- Topology ---
                self._page_persistence_barcodes(pdf)
                self._page_persistence_diagram(pdf)
                self._page_persistence_landscape(pdf)
                self._page_scale_rationale(pdf)
                
                # --- Structure (Opaque Maps) ---
                self._page_spatial_plot(
                    pdf, "tv_magnitude", "Topological Walls (Frontiers)", cmap='magma', 
                    cbar_label="Gradient Magnitude (0=Flat, 1=Boundary)", filename="map_walls.png"
                )
                self._page_spatial_plot(
                    pdf, "stability_score", "Stability Landscape (Energy)", cmap='viridis',
                    cbar_label="Stability (0=Unstable, 1=Stable Core)", filename="map_stability.png"
                )
                self._page_basins(pdf)
                self._page_robust_cores(pdf)
                
                # --- Domains ---
                if self.saliency_df is not None:
                    self._page_domain_saliency(pdf)
                
                self._page_final_domains(pdf) # Opaque domains
                self._page_domain_overlay(pdf) # Transparent overlay on histology (NEW)
                
                # --- Biology ---
                self._page_marker_dotplot(pdf, stability_markers)
                self._page_gradient_flow(pdf)

            self.logger.info(f"Report saved: {pdf_path}")
            self.logger.info(f"Individual images saved to: {self.viz_dir}")
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            raise

    # ========================================================================
    # PLOTTING METHODS
    # ========================================================================

    def _save_plot(self, fig, pdf, filename=None):
        """Helper to save to both PDF and PNG."""
        pdf.savefig(fig, bbox_inches='tight')
        if filename:
            fig.savefig(os.path.join(self.viz_dir, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _page_title_summary(self, pdf, stats):
        fig = plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.5, 0.9, "PERSIST Analysis Report", ha='center', fontsize=24, fontweight='bold')
        
        stats_text = (
            f"Tissue Type: {getattr(self.config, 'tissue_type', 'Unknown')}\n"
            f"Run Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"Metrics Summary:\n"
            f"────────────────\n"
            f"  • Scale Local Coherence (SLC): {stats.get('SLC', 0):.4f}  (>0.9 = Robust)\n"
            f"  • Orthogonality (TOI): {stats.get('TOI', 0):.4f}\n"
            f"  • Baseline ARI: {stats.get('Baseline_ARI', 0):.4f}\n\n"
            f"Topology Statistics:\n"
            f"────────────────────\n"
            f"  • H1 Features (Loops): {self.topo.tracked_h1_count}\n"
            f"  • Significant Cores: {np.sum(self.adata.obs.get('is_significant_core', 0))}\n"
            f"  • Mean Stability: {self.adata.obs['stability_score'].mean():.3f}"
        )
        plt.text(0.1, 0.5, stats_text, fontsize=12, family='monospace', va='center',
                 bbox=dict(facecolor='#f8f9fa', alpha=1.0, boxstyle='round,pad=1'))
        
        self._save_plot(fig, pdf, "summary_stats.png")

    def _page_histology(self, pdf):
        """Pure Histology Image."""
        fig, ax = plt.subplots(figsize=(10, 10))
        if "spatial" in self.adata.uns:
            sc.pl.spatial(self.adata, img_key="hires", alpha_img=1.0, color=None, show=False, ax=ax, title="H&E Histology")
        else:
            coords = self.adata.obsm["spatial"]
            ax.scatter(coords[:, 0], coords[:, 1], c='gray', s=10, alpha=0.6)
            ax.invert_yaxis()
            ax.set_title("Spatial Coordinates (No Image)")
        ax.axis('off')
        self._save_plot(fig, pdf, "ref_histology.png")

    def _page_persistence_barcodes(self, pdf):
        """H0/H1 Barcodes."""
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 14))
        
        h0 = self.topo.simplex_tree.persistence_intervals_in_dimension(0)
        h1 = self.topo.persistence_pairs
        
        self._plot_barcode_clean(ax0, h0, "H0: Connected Components", "orange")
        self._plot_barcode_clean(ax1, h1, "H1: Loops / Voids", "steelblue")
        
        plt.tight_layout()
        self._save_plot(fig, pdf, "topo_barcodes.png")

    def _plot_barcode_clean(self, ax, intervals, title, color):
        # Filter infinite and tiny noise
        finite = [p for p in intervals if p[1] != float('inf') and (p[1]-p[0]) > 1e-5]
        finite.sort(key=lambda x: x[1]-x[0], reverse=True)
        
        # Limit for file size, but keep high enough for completeness
        if len(finite) > 5000: 
            finite = finite[:5000]
        
        for i, (b, d) in enumerate(finite):
            ax.plot([b, d], [i, i], color=color, alpha=0.6, linewidth=1.5)
            
        ax.set_title(title, fontsize=14) # No "Top 2000" text
        ax.set_ylabel("Feature Rank")
        ax.set_xlabel("Scale (Alpha Squared)")
        ax.grid(True, alpha=0.2)
        ax.invert_yaxis()

    def _page_persistence_diagram(self, pdf):
        """Persistence Scatter."""
        try:
            import gudhi as gd
            fig, ax = plt.subplots(figsize=(10, 10))
            gd.plot_persistence_diagram(self.topo.simplex_tree.persistence(), axes=ax, legend=True)
            ax.set_title("Persistence Diagram")
            self._save_plot(fig, pdf, "topo_diagram.png")
        except: pass

    def _page_persistence_landscape(self, pdf):
        """Persistence Landscape."""
        if hasattr(self.topo, 'landscapes') and self.topo.landscapes is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(np.mean(self.topo.landscapes, axis=0), color='teal', linewidth=2)
            ax.set_title("Topological Landscape Signature")
            ax.set_xlabel("Landscape Index")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            self._save_plot(fig, pdf, "topo_landscape.png")

    def _page_scale_rationale(self, pdf):
        """Scale selection."""
        if not hasattr(self.tv, 'scale_history') or not self.tv.scale_history: return
        scales = sorted(self.tv.scale_history.keys())
        mean_stab = [np.mean(self.tv.scale_history[s]) for s in scales]
        
        try: weights = self.topo.compute_scale_weights(np.array(scales))
        except: weights = np.ones(len(scales))

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar([str(int(s)) for s in scales], weights, color='tab:blue', alpha=0.6, label='Weight')
        ax1.set_ylabel('Topological Weight', color='tab:blue')
        
        ax2 = ax1.twinx()
        ax2.plot([str(int(s)) for s in scales], mean_stab, color='tab:red', marker='o', linewidth=2, label='Stability')
        ax2.set_ylabel('Mean Stability', color='tab:red')
        
        ax1.set_title("Scale Selection Rationale")
        self._save_plot(fig, pdf, "scale_rationale.png")

    def _page_spatial_plot(self, pdf, obs_key, title, cmap, cbar_label, filename):
        """Solid Opaque Map."""
        if obs_key not in self.adata.obs: return
        fig, ax = plt.subplots(figsize=(10, 10))
        coords = self.adata.obsm["spatial"]
        
        # Alpha=1.0 for solid map, size=18 for continuity
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=self.adata.obs[obs_key], 
                        cmap=cmap, s=18, alpha=1.0, edgecolors='none')
        
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title(title, fontsize=16)
        
        cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label(cbar_label, fontsize=10) # Descriptive legend
        
        self._save_plot(fig, pdf, filename)

    def _page_basins(self, pdf):
        """Basin Contours."""
        fig, ax = plt.subplots(figsize=(10, 10))
        coords = self.adata.obsm["spatial"]
        ax.scatter(coords[:, 0], coords[:, 1], c='#e0e0e0', s=15, alpha=1.0) # Solid background
        
        walls = np.nan_to_num(self.adata.obs["tv_magnitude"].values)
        if len(walls) > 0:
            grid_x, grid_y = np.mgrid[coords[:,0].min():coords[:,0].max():300j, coords[:,1].min():coords[:,1].max():300j]
            grid = griddata(coords, walls, (grid_x, grid_y), method='linear', fill_value=0)
            grid = gaussian_filter(grid, sigma=1)
            ax.contour(grid_x, grid_y, grid, levels=[np.percentile(walls, 80)], colors='#D32F2F', linewidths=2.0)
            
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title("Basin Decomposition (Red = Walls)")
        self._save_plot(fig, pdf, "map_basins.png")

    def _page_robust_cores(self, pdf):
        """Solid Cores."""
        fig, ax = plt.subplots(figsize=(10, 10))
        coords = self.adata.obsm["spatial"]
        if "is_significant_core" in self.adata.obs:
            mask = self.adata.obs["is_significant_core"].values
            # Non-core
            ax.scatter(coords[~mask, 0], coords[~mask, 1], c='#e0e0e0', s=15, alpha=1.0)
            # Core
            ax.scatter(coords[mask, 0], coords[mask, 1], c='#B71C1C', s=18, alpha=1.0)
            ax.set_title(f"Robust Stability Cores ({mask.sum()} spots)")
        ax.invert_yaxis()
        ax.axis('off')
        self._save_plot(fig, pdf, "map_cores.png")

    def _page_domain_saliency(self, pdf):
        """Saliency Plot with Diagonal."""
        if self.saliency_df is None or self.saliency_df.empty: return
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sc = ax.scatter(self.saliency_df['peak_stability'], self.saliency_df['saliency'], 
                        s=self.saliency_df['size']/3, c=self.saliency_df['saliency'], 
                        cmap='plasma', alpha=0.8, edgecolors='k')
        
        # Add Reference Line (y=x) if units match, but here y = max - mean.
        # Max Saliency possible is Peak itself (if Mean=0).
        # So points should lie below y=x line.
        lims = [0, 1.0]
        ax.plot(lims, lims, 'k--', alpha=0.3, label="Max Theoretical Saliency")
        
        for _, row in self.saliency_df.head(5).iterrows():
            ax.annotate(f"D{row['domain']}", (row['peak_stability'], row['saliency']))
            
        ax.set_xlabel("Peak Stability Score (Max)")
        ax.set_ylabel("Saliency (Max - Mean)")
        ax.set_title("Domain Saliency Analysis")
        plt.colorbar(sc, ax=ax, label="Saliency Magnitude")
        ax.grid(True, linestyle='--', alpha=0.3)
        self._save_plot(fig, pdf, "domain_saliency.png")

    def _page_final_domains(self, pdf):
        """Solid Domains Map."""
        fig, ax = plt.subplots(figsize=(12, 10))
        coords = self.adata.obsm["spatial"]
        key = "domain_annotation" if "domain_annotation" in self.adata.obs else "stability_domain"
        
        import matplotlib.cm as cm
        domains = self.adata.obs[key].unique()
        colors = cm.tab20(np.linspace(0, 1, len(domains)))
        
        for i, d in enumerate(domains):
            mask = self.adata.obs[key] == d
            # Solid Alpha = 1.0
            ax.scatter(coords[mask, 0], coords[mask, 1], color=colors[i], s=18, alpha=1.0, label=d)
            
        ax.invert_yaxis()
        ax.axis('off')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Domains")
        ax.set_title("Final Topological Domains (Solid Map)")
        self._save_plot(fig, pdf, "map_domains_solid.png")

    def _page_domain_overlay(self, pdf):
        """NEW: Transparent Overlay on Histology."""
        # Only works if image exists
        if "spatial" not in self.adata.uns: return

        fig, ax = plt.subplots(figsize=(12, 10))
        key = "domain_annotation" if "domain_annotation" in self.adata.obs else "stability_domain"
        
        # Scanpy plot with transparency
        sc.pl.spatial(
            self.adata, 
            color=key, 
            alpha_img=0.9,      # High background visibility
            alpha=0.6,          # Transparent spots
            size=1.3,
            show=False, 
            ax=ax, 
            title="Domain Overlay on Histology",
            legend_loc="right margin"
        )
        
        ax.set_xlabel("")
        ax.set_ylabel("")
        self._save_plot(fig, pdf, "map_domains_overlay.png")

    def _page_marker_dotplot(self, pdf, markers, reference_markers=None):
        """
        Page 13: Marker Dotplot (Sorted & Robust).
        Sorts categories to reveal diagonal structure and uses robust figure handling.
        """
        self.logger.info("Generating Biological Marker Dotplot...")

        # --- 1. PREPARE MARKERS ---
        valid_markers = {}
        
        # Helper to filter genes
        def get_valid(genes):
            return [g for g in genes if g in self.adata.var_names]

        # Add Stability Markers
        if markers:
            for group, genes in markers.items():
                v = get_valid(genes)
                if v: valid_markers[group] = v
        
        # Add Reference Markers
        if reference_markers:
            v = get_valid(reference_markers)
            if v: valid_markers["Reference"] = v

        if not valid_markers:
            self.logger.warning("Skipping Dotplot: No valid genes found.")
            return

        # --- 2. PREPARE DATA & SORTING (The Fix) ---
        key = "domain_annotation" if "domain_annotation" in self.adata.obs else "stability_domain"
        
        # Safety: Fill NaNs
        if key in self.adata.obs:
            if self.adata.obs[key].isnull().any():
                self.adata.obs[key] = self.adata.obs[key].astype(str).fillna("Unassigned")
            
            # Ensure categorical
            self.adata.obs[key] = self.adata.obs[key].astype('category')

            # This forces Scanpy to group rows alphabetically/numerically
            # rather than randomly or by cluster ID appearance.
            try:
                # Get unique categories and sort them naturally
                categories = sorted(self.adata.obs[key].unique().astype(str))
                self.adata.obs[key] = self.adata.obs[key].cat.reorder_categories(categories)
            except Exception as e:
                self.logger.warning(f"Could not sort dotplot categories: {e}")

        # --- 3. PLOT ("The Old Reliable" Way) ---
        try:
            # Step A: Explicitly create a new figure context
            plt.figure() 
            
            # Step B: Run Scanpy
            sc.pl.dotplot(
                self.adata, 
                valid_markers, 
                groupby=key, 
                standard_scale='var', 
                show=False  # Keep plot open
            )
            
            # Step C: Grab the current figure immediately
            fig = plt.gcf()
            
            # Step D: Resize dynamically based on content
            n_genes = sum(len(v) for v in valid_markers.values())
            n_domains = len(self.adata.obs[key].unique())
            
            # Width: ~0.35 inches per gene, Height: ~0.5 inches per domain
            # Min dimensions (8x5) ensure small plots don't look crushed
            fig.set_size_inches(max(8, n_genes * 0.35), max(5, n_domains * 0.5))
            
            # Step E: Save
            plt.suptitle("Biological Validation", y=1.02, fontsize=16)
            self._save_plot(fig, pdf, "bio_dotplot.png")
            self.logger.info(f"Marker dotplot saved successfully (Sorted {n_domains} domains).")
            
        except Exception as e:
            self.logger.error(f"DOTPLOT FAILED: {str(e)}")
            plt.close()

    def _page_gradient_flow(self, pdf):
        """Gradient Flow."""
        fig, ax = plt.subplots(figsize=(10, 10))
        coords = self.adata.obsm["spatial"]
        # Background slightly transparent to see arrows
        ax.scatter(coords[:, 0], coords[:, 1], c=self.adata.obs["stability_score"], 
                   cmap='viridis', alpha=0.4, s=15)
        
        # Vector calc
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=5).fit(coords)
        u, v = np.zeros(len(coords)), np.zeros(len(coords))
        stab = self.adata.obs["stability_score"].values
        ind = nbrs.kneighbors(coords, return_distance=False)
        
        for i in range(len(coords)):
            for n_idx in ind[i][1:]:
                d = coords[n_idx] - coords[i]
                s = stab[n_idx] - stab[i]
                norm = np.linalg.norm(d)
                if norm > 0:
                    u[i] += (s * d[0]) / norm
                    v[i] += (s * d[1]) / norm
                
        step = max(1, len(coords)//500)
        ax.quiver(coords[::step, 0], coords[::step, 1], u[::step], v[::step], scale=20, headwidth=3)
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title("Stability Gradient Flow")
        self._save_plot(fig, pdf, "bio_gradients.png")

    def _compute_domain_saliency(self):
        if 'stability_domain' not in self.adata.obs: return None
        stats = []
        for d in self.adata.obs['stability_domain'].unique():
            mask = self.adata.obs['stability_domain'] == d
            s = self.adata.obs.loc[mask, 'stability_score']
            if len(s) > 0:
                stats.append({'domain': d, 'saliency': s.max()-s.mean(), 'peak_stability': s.max(), 'size': mask.sum()})
        return pd.DataFrame(stats)


# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

class PERSISTPipeline:
    """
    Main PERSIST pipeline orchestrator.
    
    Coordinates the five phases of analysis:
    1. Preprocessing & mesh construction
    2. Topological scale analysis
    3. Multiscale total variation
    4. Basin decomposition
    5. Validation & visualization
    """
    
    def __init__(
        self,
        adata: sc.AnnData,
        config: Optional[PERSISTConfig] = None,
        output_dir: str = ".",
        random_state: int = 42
    ) -> None:
        """
        Initialize the PERSIST pipeline.
        
        Args:
            adata: Input AnnData object with spatial coordinates
            config: PERSIST configuration (uses defaults if None)
            output_dir: Directory for saving results
            random_state: Random seed for reproducibility
            
        Raises:
            ValueError: If input data is invalid
        """
        self.config: PERSISTConfig = config or PERSISTConfig()
        self.adata: sc.AnnData = adata
        self.output_dir: str = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Setup logging
        self.logger: logging.Logger = setup_logger(output_dir)
        
        # Set random seeds for reproducibility
        self.config.random_seed = random_state
        np.random.seed(random_state)
        if GPU_AVAILABLE:
            try:
                cp.random.seed(random_state)
            except Exception as e:
                self.logger.warning(f"Failed to set GPU random seed: {e}")
        
        # Initialize pipeline components
        self.preprocessor: Optional[SpatialPreprocessor] = None
        self.topology: Optional[TopologicalScaleAnalyzer] = None
        self.tv_operator: Optional[MultiscaleTVOperator] = None
        self.basin_decomposer: Optional[TopographicBasinDecomposer] = None
        self.metrics: Optional[PERSISTMetrics] = None
        self.visualizer: Optional[PERSISTVisualizer] = None
        
        # Validate input data
        self._validate_input()
    
    def _validate_input(self) -> None:
        """Validate input AnnData structure and content."""
        self.logger.info("Validating input data...")
        
        if self.adata is None:
            raise ValueError("Input AnnData is None")
        
        if self.adata.shape[0] == 0 or self.adata.shape[1] == 0:
            raise ValueError("Empty AnnData object")
        
        if 'spatial' not in self.adata.obsm:
            raise ValueError(
                "Missing spatial coordinates in adata.obsm['spatial']. "
                "Please load spatial data first."
            )
        
        self.logger.info(
            f"Input data validated: "
            f"{self.adata.shape[0]} spots, {self.adata.shape[1]} genes"
        )
    
    def run(
        self,
        n_top_genes: int = 2000,
        n_scales: int = 5,
        tissue_type: Optional[str] = None,
        reference_markers: Optional[List[str]] = None,
        stability_markers: Optional[Dict[str, List[str]]] = None
    ) -> None:
        """
        Execute the complete PERSIST pipeline.
        
        Args:
            n_top_genes: Number of top genes to select
            n_scales: Number of scales for multiscale analysis
            tissue_type: Tissue type for schema lookup
            reference_markers: Marker genes for biological validation
            stability_markers: Stability-associated markers for annotation
            
        Raises:
            RuntimeError: If any pipeline phase fails
            ValueError: If parameters are invalid
        """
        self.logger.info("=" * 60)
        self.logger.info("PERSIST PIPELINE STARTED")
        self.logger.info("=" * 60)
        self.logger.info(f"Tissue type: {tissue_type or 'Not specified'}")
        self.logger.info(f"Number of scales: {n_scales}")
        self.logger.info(f"Top genes: {n_top_genes}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        try:
            # --- Tissue Schema Integration ---
            if tissue_type and tissue_type in TISSUE_SCHEMA:
                self.logger.info(f"Using knowledge base for: {tissue_type}")
                schema = TISSUE_SCHEMA[tissue_type]
                
                # Load defaults if not manually overridden
                if reference_markers is None:
                    reference_markers = schema.get("reference_markers")
                if stability_markers is None:
                    stability_markers = schema.get("stability_markers")
                    
                self.logger.info(
                    f"Loaded {len(stability_markers or {})} stability markers, "
                    f"{len(reference_markers or [])} reference markers"
                )
            elif tissue_type is not None:
                self.logger.warning(
                    f"Tissue '{tissue_type}' not found in schema. Using generic mode."
                )
            
            # Phase 0-1: Preprocessing and mesh construction
            self.logger.info("\n" + "=" * 40)
            self.logger.info("PHASES 0-1: PREPROCESSING & MESH CONSTRUCTION")
            self.logger.info("=" * 40)
            
            self.preprocessor = SpatialPreprocessor(
                self.adata, self.config, self.logger
            )
            self.adata, mesh, alpha_wrapper, st, median_dist = (
                self.preprocessor.run(
                    n_top_genes=n_top_genes,
                    tissue_type=tissue_type or "human_lymph_node"
                )
            )
            
            # Phase 2: Topological scale analysis
            self.logger.info("\n" + "=" * 40)
            self.logger.info("PHASE 2: TOPOLOGICAL SCALE ANALYSIS")
            self.logger.info("=" * 40)
            
            self.topology = TopologicalScaleAnalyzer(
                st, median_dist, alpha_wrapper, mesh, self.config, self.logger
            )
            self.topology.run()
            
            # Determine optimal scales based on persistence
            optimal_scales = self.topology.compute_optimal_scales(n_scales)
            
            # Phase 3: Multiscale total variation
            self.logger.info("\n" + "=" * 40)
            self.logger.info("PHASE 3: MULTISCALE TOTAL VARIATION")
            self.logger.info("=" * 40)
            
            self.tv_operator = MultiscaleTVOperator(
                self.adata, mesh, alpha_wrapper, self.config, self.logger
            )
            self.tv_operator.run(optimal_scales, self.topology)
            self.tv_operator.compute_robustness()
            
            # Phase 4: Basin decomposition
            self.logger.info("\n" + "=" * 40)
            self.logger.info("PHASE 4: BASIN DECOMPOSITION")
            self.logger.info("=" * 40)
            
            self.basin_decomposer = TopographicBasinDecomposer(
                self.adata, median_dist, mesh, self.config, self.logger
            )
            self.basin_decomposer.run()
            
            # Phase 5: Validation and visualization
            self.logger.info("\n" + "=" * 40)
            self.logger.info("PHASE 5: VALIDATION & VISUALIZATION")
            self.logger.info("=" * 40)
            
            # 5a. Metrics
            self.metrics = PERSISTMetrics(
                self.adata, self.tv_operator, self.config,
                self.output_dir, self.logger
            )
            stats = self.metrics.run(
                reference_markers=reference_markers,
                stability_markers=stability_markers,
                baseline_key='leiden'
            )
            
            # 5b. Visualization
            self.visualizer = PERSISTVisualizer(
                self.adata, mesh, self.topology, self.tv_operator,
                self.config, self.output_dir, self.logger
            )
            self.visualizer.run(
                stats=stats,
                reference_markers=reference_markers,
                stability_markers=stability_markers
            )
            
            # Save final state
            self._save_state()
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("PERSIST PIPELINE COMPLETE")
            self.logger.info("=" * 60)
            self.logger.info(f"Results saved to: {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed at phase: {e}")
            self.logger.error("Pipeline terminated with errors.")
            raise RuntimeError(f"PERSIST pipeline failed: {e}")
    
    def _save_state(self) -> None:
        """Save pipeline state for reproducibility and analysis."""
        filename = os.path.join(self.output_dir, "persist_final_state.npz")
        
        try:
            # Collect key results
            scale_history = (
                self.tv_operator.scale_history if self.tv_operator else {}
            )
            stability_scores = (
                self.adata.obs.get('stability_score', []).values
                if hasattr(self.adata.obs, 'get') else []
            )
            
            # Save compressed numpy archive
            np.savez_compressed(
                filename,
                scale_history=scale_history,
                stability_scores=stability_scores,
                config=self.config.to_dict() if hasattr(self.config, 'to_dict') else {}
            )
            
            self.logger.info(f"Pipeline state saved to {filename}")
        except Exception as e:
            self.logger.warning(f"Failed to save pipeline state: {e}")


# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the PERSIST pipeline.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description=(
            "PERSIST: Persistence-driven Spatial Identification of Stability Topologies\n"
            "A topological data analysis pipeline for spatial transcriptomics that combines\n"
            "persistent homology, multiscale total variation, and watershed segmentation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Core Data Arguments ---
    parser.add_argument(
        "--tissue_type",
        type=str,
        required=True,
        choices=list(TISSUE_SCHEMA.keys()),
        help="Type of tissue to analyze. Determines marker selection and topology strategy."
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help=(
            "Optional local path (.h5ad or Visium dir). "
            "If not provided, pulls the standard dataset directly from Visium SGE."
        )
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory to save results. "
            "If empty, defaults to './persist_output_{tissue_type}'"
        )
    )
    
    # --- Analysis Configuration ---
    parser.add_argument(
        "--n_top_genes",
        type=int,
        default=2000,
        help="Number of highly variable genes to use."
    )
    
    parser.add_argument(
        "--n_scales",
        type=int,
        default=5,
        help="Number of alpha-complex scales to analyze."
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    
    # --- Advanced Overrides ---
    parser.add_argument(
        "--min_k",
        type=int,
        default=None,
        help="Override default neighbor count for graph construction."
    )
    
    parser.add_argument(
        "--auto_bridge",
        action='store_true',
        help="Force enable graph bridging (useful for fragmented tissues)."
    )
    
    # --- Logging Options ---
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for the pipeline."
    )
    
    return parser.parse_args()


def load_data(args: argparse.Namespace, logger: logging.Logger) -> sc.AnnData:
    """
    Load data logic:
    1. If --data_path is provided, load local file.
    2. If NOT provided, pull standard Visium dataset via Scanpy.
    
    Args:
        args: Command-line arguments
        logger: Logger instance
        
    Returns:
        sc.AnnData: Loaded spatial transcriptomics data
        
    Raises:
        FileNotFoundError: If local data path doesn't exist
        ValueError: If data format is unsupported
    """
    # 1. Local Load
    if args.data_path and os.path.exists(args.data_path):
        logger.info(f"Loading local dataset from: {args.data_path}")
        
        if args.data_path.endswith('.h5ad'):
            adata = sc.read_h5ad(args.data_path)
        elif os.path.isdir(args.data_path):
            adata = sc.read_visium(args.data_path)
        else:
            raise ValueError(f"Unsupported file format: {args.data_path}")
        
        # Standardize spatial key
        if 'spatial' not in adata.obsm:
            if 'X_spatial' in adata.obsm:
                adata.obsm['spatial'] = adata.obsm['X_spatial']
                logger.info("Mapped .obsm['X_spatial'] to .obsm['spatial']")
            else:
                raise ValueError(
                    "No spatial coordinates found in adata.obsm. "
                    "Expected either 'spatial' or 'X_spatial'."
                )
        
        logger.info(
            f"Local data loaded: {adata.shape[0]} spots, {adata.shape[1]} genes"
        )
        return adata
    
    # 2. Pull from Visium (Standard Datasets)
    logger.info(
        f"No local path provided. Pulling standard Visium dataset for: {args.tissue_type}"
    )
    
    try:
        if args.tissue_type == "human_lymph_node":
            # Standard 10x Human Lymph Node
            adata = sc.datasets.visium_sge(sample_id="V1_Human_Lymph_Node")
        elif args.tissue_type == "mouse_brain":
            # Standard 10x Mouse Brain (Sagittal Posterior)
            adata = sc.datasets.visium_sge(sample_id="V1_Adult_Mouse_Brain")
        elif args.tissue_type == "breast_cancer":
            # Standard 10x Breast Cancer (Block A)
            adata = sc.datasets.visium_sge(sample_id="V1_Breast_Cancer_Block_A_Section_1")
        else:
            raise ValueError(
                f"No standard Visium ID defined for {args.tissue_type}. "
                "Please provide --data_path."
            )
        
        logger.info(
            f"Standard dataset loaded: {adata.shape[0]} spots, {adata.shape[1]} genes"
        )
        return adata
        
    except Exception as e:
        logger.error(f"Failed to load standard dataset: {e}")
        raise


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """
    Main execution function for PERSIST pipeline.
    
    Handles:
    1. Argument parsing
    2. Data loading
    3. Pipeline execution
    4. Error handling and logging
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"./persist_output_{args.tissue_type}"
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Setup logging
    logger = setup_logger(args.output_dir, args.log_level)
    
    # Print pipeline header
    header = f"""
        ╔══════════════════════════════════════════════════════════╗
        ║                 PERSIST PIPELINE v1.0                    ║
        ╠══════════════════════════════════════════════════════════╣
        ║  Tissue: {args.tissue_type:<35}                    ║
        ║  Output: {args.output_dir:<35}                 ║
        ║  Random Seed: {args.seed:<32}                  ║
        ╚══════════════════════════════════════════════════════════╝
    """
    
    print(header)
    logger.info("PERSIST Pipeline Starting")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load data
        adata = load_data(args, logger)
        logger.info(f"Data loaded successfully: {adata.shape}")
        
        # Basic preprocessing check
        if np.max(adata.X) > 100:
            logger.info("Raw counts detected. Normalizing...")
            adata.var_names_make_unique()
            sc.pp.calculate_qc_metrics(adata, inplace=True)
            sc.pp.filter_genes(adata, min_cells=3)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            logger.info("Normalization complete")
        else:
            logger.info("Data appears pre-normalized. Skipping standard preprocessing.")
        
        # Configure pipeline
        config = PERSISTConfig(random_seed=args.seed)
        if args.min_k:
            config.pruning_neighbors = args.min_k
            logger.info(f"Overriding min_k to {args.min_k}")
        
        # Create and run pipeline
        pipeline = PERSISTPipeline(
            adata,
            config=config,
            output_dir=args.output_dir,
            random_state=args.seed
        )
        
        pipeline.run(
            n_top_genes=args.n_top_genes,
            n_scales=args.n_scales,
            tissue_type=args.tissue_type,
        )
        
        # Success message
        success_msg = f"""
        PERSIST Pipeline Completed Successfully!
        
        Results saved to: {args.output_dir}
        
        Generated files:
          - persist_pipeline.log        : Complete pipeline log
          - persist_metrics.json        : Quantitative metrics
          - persist_final_state.npz     : Pipeline state
          - visualizations/             : All generated figures
        """
        
        print(success_msg)
        logger.info("Pipeline completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        print(f"ERROR: Data file not found: {e}")
        sys.exit(1)
        
    except ValueError as e:
        logger.error(f"Invalid input or parameter: {e}")
        print(f"ERROR: Invalid input: {e}")
        sys.exit(1)
        
    except RuntimeError as e:
        logger.error(f"Pipeline runtime error: {e}")
        print(f"ERROR: Pipeline failed: {e}")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback_str = traceback.format_exc()
        logger.error(f"Traceback:\n{traceback_str}")
        print(f"UNEXPECTED ERROR: {e}")
        print("Check the log file for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
