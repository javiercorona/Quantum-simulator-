from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.visualization import (
    plot_bloch_vector,
    plot_state_city,
    plot_state_qsphere,
    plot_state_paulivec,
    plot_histogram,
    plot_state_hinton,
    plot_state_city,
    circuit_drawer,
)
from qiskit.result import Result
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from typing import Optional, Union, Dict, List
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuantumViz")

# Optional: Use plotly for interactive plots (if installed)
try:
    import plotly
    USE_PLOTLY = True
except ImportError:
    USE_PLOTLY = False
    logger.warning("Plotly not installed. Using matplotlib for visualizations.")

class QuantumVisualizer:
    """Enhanced quantum circuit and state visualizer with multiple backends."""
    
    def __init__(self, result: Optional[Result] = None, circuit: Optional[QuantumCircuit] = None):
        self.result = result
        self.circuit = circuit
        self.backend = None
        self._validate_inputs()

    def _validate_inputs(self):
        """Check if either a circuit or result is provided."""
        if self.result is None and self.circuit is None:
            raise ValueError("Either a QuantumCircuit or a Result object must be provided.")

    def visualize_circuit(
        self,
        output_format: str = "mpl",
        save_path: str = "circuit.png",
        show: bool = True,
        interactive: bool = False,
        **kwargs
    ) -> Optional[plt.Figure]:
        """Render and save a visualization of the quantum circuit.
        
        Args:
            output_format (str): "mpl" (matplotlib), "latex", "text", or "plotly" (if installed).
            save_path (str): Path to save the image (PDF/SVG/PNG supported).
            show (bool): Whether to display the plot.
            interactive (bool): If True and plotly is installed, generates interactive plot.
            **kwargs: Additional args for circuit_drawer.
            
        Returns:
            Figure if successful, else None.
        """
        if self.circuit is None:
            logger.error("No circuit provided for visualization.")
            return None

        try:
            if interactive and USE_PLOTLY and output_format == "plotly":
                fig = circuit_drawer(self.circuit, output="plotly", **kwargs)
                if save_path:
                    fig.write_image(save_path)
                if show:
                    fig.show()
                return fig
            else:
                fig = circuit_drawer(self.circuit, output=output_format, **kwargs)
                if save_path:
                    fig.savefig(save_path, bbox_inches="tight")
                if show:
                    plt.show()
                return fig
        except Exception as e:
            logger.error(f"Circuit visualization failed: {e}")
            return None

    def bloch_visualization(
        self,
        qubits: Optional[List[int]] = None,
        save_dir: str = "bloch_spheres",
        show: bool = True,
        style: Optional[Dict] = None,
    ) -> bool:
        """Visualize Bloch spheres for selected qubits.
        
        Args:
            qubits (List[int]): Qubit indices to visualize (default: all).
            save_dir (str): Directory to save images.
            show (bool): Whether to display plots.
            style (Dict): Custom Bloch sphere styling (e.g., {"size": [500, 500]}).
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if self.result is None:
            logger.error("No result provided for Bloch visualization.")
            return False

        try:
            state = self.result.get_statevector()
            Path(save_dir).mkdir(exist_ok=True)

            if qubits is None:
                qubits = range(len(state))

            for i in qubits:
                if i >= len(state):
                    logger.warning(f"Qubit index {i} out of range.")
                    continue

                bloch_fig = plot_bloch_vector(state[i], title=f"Qubit {i}", **(style or {}))
                save_path = os.path.join(save_dir, f"bloch_q{i}.png")
                bloch_fig.savefig(save_path)
                if show:
                    plt.show()
                plt.close(bloch_fig)

            return True
        except Exception as e:
            logger.error(f"Bloch visualization failed: {e}")
            return False

    def state_visualizations(
        self,
        methods: List[str] = ["city", "qsphere", "paulivec", "hinton"],
        save_dir: str = "state_plots",
        show: bool = True,
        **kwargs
    ) -> Dict[str, Optional[plt.Figure]]:
        """Generate multiple state visualizations at once.
        
        Args:
            methods (List[str]): Visualization types ("city", "qsphere", "paulivec", "hinton").
            save_dir (str): Directory to save plots.
            show (bool): Whether to display plots.
            **kwargs: Additional args for plotting functions.
            
        Returns:
            Dict[str, Optional[plt.Figure]]: Mapping of method to generated figure.
        """
        if self.result is None:
            logger.error("No result provided for state visualization.")
            return {method: None for method in methods}

        Path(save_dir).mkdir(exist_ok=True)
        state = self.result.get_statevector()
        figures = {}

        for method in methods:
            try:
                if method == "city":
                    fig = plot_state_city(state, **kwargs)
                elif method == "qsphere":
                    fig = plot_state_qsphere(state, **kwargs)
                elif method == "paulivec":
                    fig = plot_state_paulivec(state, **kwargs)
                elif method == "hinton":
                    fig = plot_state_hinton(state, **kwargs)
                else:
                    logger.warning(f"Unknown visualization method: {method}")
                    continue

                save_path = os.path.join(save_dir, f"{method}_plot.png")
                fig.savefig(save_path)
                if show:
                    plt.show()
                figures[method] = fig
            except Exception as e:
                logger.error(f"{method} visualization failed: {e}")
                figures[method] = None

        return figures

    def histogram_visualization(
        self,
        save_path: str = "measurement_histogram.png",
        show: bool = True,
        color: Optional[str] = None,
        **kwargs
    ) -> Optional[plt.Figure]:
        """Plot measurement counts as a histogram.
        
        Args:
            save_path (str): Path to save the histogram.
            show (bool): Whether to display the plot.
            color (str): Bar color (e.g., "skyblue").
            **kwargs: Additional args for plot_histogram.
            
        Returns:
            plt.Figure if successful, else None.
        """
        if self.result is None:
            logger.error("No result provided for histogram.")
            return None

        try:
            counts = self.result.get_counts()
            fig = plot_histogram(counts, color=color, **kwargs)
            fig.savefig(save_path, bbox_inches="tight")
            if show:
                plt.show()
            return fig
        except Exception as e:
            logger.error(f"Histogram visualization failed: {e}")
            return None

# ===== Example Usage =====
if __name__ == "__main__":
    # Create a simple circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    # Simulate
    backend = Aer.get_backend("qasm_simulator")
    result = execute(qc, backend, shots=1024).result()

    # Initialize visualizer
    visualizer = QuantumVisualizer(result=result, circuit=qc)

    # Generate visualizations
    visualizer.visualize_circuit(save_path="circuit.pdf", show=True)
    visualizer.bloch_visualization(qubits=[0, 1], save_dir="bloch_plots")
    visualizer.state_visualizations(methods=["city", "qsphere"], save_dir="state_viz")
    visualizer.histogram_visualization(color="purple", save_path="counts_histogram.png")
