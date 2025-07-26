import numpy as np
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.circuit import Parameter, Gate, Instruction, ParameterExpression
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import Statevector, Operator, state_fidelity, entropy
from qiskit.visualization import (plot_histogram, plot_bloch_multivector,
                                 plot_state_city, plot_state_qsphere,
                                 plot_state_paulivec, circuit_drawer)
from qiskit.providers.aer import AerSimulator, noise
from qiskit.providers.fake_provider import FakeWashington
from qiskit.qpy import dump, load
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from typing import List, Dict, Union, Optional, Callable, Any, Tuple
import matplotlib.pyplot as plt
import json
import logging
import asyncio
import concurrent.futures
import sys
import os
from datetime import datetime
from functools import lru_cache
from enum import Enum, auto
import warnings
import pickle
from pathlib import Path

# Configure enhanced logging
logging.basicConfig(
    filename='quantum_simulator.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)

class BackendType(Enum):
    STATEVECTOR = auto()
    QASM = auto()
    UNITARY = auto()
    GPU = auto()
    RUNTIME = auto()

class VisualizationType(Enum):
    BLOCH = auto()
    CITY = auto()
    QSPHERE = auto()
    PAULI = auto()
    HISTOGRAM = auto()
    CIRCUIT = auto()

class QuantumSimulator:
    """Production-grade quantum simulator with complete feature set"""
    def __init__(self, max_qubits: int = 20, backend_type: BackendType = BackendType.STATEVECTOR,
                 unitary_display_limit: int = 5):
        self._validate_environment()
        self.max_qubits = max_qubits
        self.backend_type = backend_type
        self._unitary_display_limit = unitary_display_limit
        self._init_backend()
        self.supported_gates = self._init_gate_map()
        self._parameters = {}
        self._custom_gates = {}
        self._noise_model = None
        self._noise_model_cache = {}
        self._runtime_service = None
        self._active_sessions = {}
        self._executor = concurrent.futures.ThreadPoolExecutor()
        self._session_cache = {}
        self._visualizers = self._init_visualizers()
        self._calibration_matrices = {}
        logger.info(f"Simulator initialized with {backend_type.name} backend")

    def __del__(self):
        """Cleanup resources on deletion"""
        self.shutdown()

    def shutdown(self):
        """Properly cleanup resources"""
        self._executor.shutdown(wait=True)
        for session in self._active_sessions.values():
            try:
                session.close()
            except Exception as e:
                logger.warning(f"Error closing session: {str(e)}")
        logger.info("Simulator shutdown complete")

    # === Enhanced Custom Gate Handling ===
    def add_custom_gate(self,
                       name: str,
                       gate: Union[Gate, Instruction, np.ndarray],
                       qubits: List[int],
                       clbits: Optional[List[int]] = None):
        """Register custom gate with proper unitary handling"""
        try:
            if isinstance(gate, np.ndarray):
                # Validate matrix before creating gate
                if not (gate.ndim == 2 and gate.shape[0] == gate.shape[1]):
                    raise ValueError("Gate matrix must be square")
                if not np.isclose(np.linalg.det(gate @ gate.conj().T), 1):
                    warnings.warn("Gate matrix may not be unitary")
                gate = UnitaryGate(gate, label=name)
            elif not isinstance(gate, (Gate, Instruction)):
                raise TypeError("Gate must be Gate, Instruction, or numpy array")
            
            self._custom_gates[name] = (gate, qubits, clbits or [])
            logger.info(f"Added custom gate: {name} ({gate.__class__.__name__})")
        except Exception as e:
            logger.error(f"Failed to add custom gate: {str(e)}")
            raise

    # === Session Management with Parameter Expressions ===
    def save_session(self, path: Union[str, Path]):
        """Save session with full parameter expression support"""
        path = Path(path)
        path.mkdir(exist_ok=True)
        
        try:
            # Save custom gates
            gate_dir = path / "custom_gates"
            gate_dir.mkdir(exist_ok=True)
            for gate_name, (gate, qubits, clbits) in self._custom_gates.items():
                with open(gate_dir / f"{gate_name}.qpy", 'wb') as f:
                    dump(gate, f)

            # Serialize parameter expressions
            param_expressions = {}
            for name, param in self._parameters.items():
                if isinstance(param, ParameterExpression):
                    param_expressions[name] = {
                        'expr': str(param._symbol_expr),
                        'params': list(param.parameters)
                    }

            session_data = {
                'parameters': list(self._parameters.keys()),
                'parameter_expressions': param_expressions,
                'custom_gates': {k: (qubits, clbits) for k, (_, qubits, clbits) in self._custom_gates.items()},
                'backend': self.backend_type.name,
                'unitary_limit': self._unitary_display_limit,
                'calibration_matrices': {
                    k: v.tolist() for k, v in self._calibration_matrices.items()
                }
            }
            
            with open(path / 'session.json', 'w') as f:
                json.dump(session_data, f, indent=2)
                
            logger.info(f"Session saved to {path}")
        except Exception as e:
            logger.error(f"Session save failed: {str(e)}")
            raise

    def load_session(self, path: Union[str, Path]):
        """Load session with parameter expression reconstruction"""
        path = Path(path)
        
        try:
            with open(path / 'session.json', 'r') as f:
                session_data = json.load(f)
            
            # Reinitialize core state
            self._unitary_display_limit = session_data.get('unitary_limit', 5)
            self.switch_backend(BackendType[session_data['backend']])
            
            # Recreate basic parameters
            self._parameters = {name: Parameter(name) for name in session_data['parameters']}
            
            # Reconstruct parameter expressions
            for name, expr_data in session_data.get('parameter_expressions', {}).items():
                expr = eval(expr_data['expr'], None, {
                    p.name: self._parameters[p.name] for p in expr_data['params']
                })
                self._parameters[name] = expr
            
            # Load custom gates
            gate_dir = path / "custom_gates"
            self._custom_gates = {}
            for gate_name, (qubits, clbits) in session_data['custom_gates'].items():
                with open(gate_dir / f"{gate_name}.qpy", 'rb') as f:
                    gate = load(f)[0]
                self._custom_gates[gate_name] = (gate, qubits, clbits)
            
            # Load calibration matrices
            self._calibration_matrices = {
                k: np.array(v) for k, v in session_data.get('calibration_matrices', {}).items()
            }
            
            logger.info(f"Session loaded from {path}")
        except Exception as e:
            logger.error(f"Session load failed: {str(e)}")
            raise

    # === Enhanced Visualization ===
    def visualize(self,
                 data: Union[Statevector, Dict[str, int], QuantumCircuit],
                 style: VisualizationType = VisualizationType.HISTOGRAM,
                 save_path: Optional[Union[str, Path]] = None,
                 interactive: bool = False,
                 **kwargs) -> Optional[Union[plt.Figure, Any]]:
        """
        Robust visualization with error handling
        Args:
            data: Input to visualize
            style: Visualization type
            save_path: Optional save path
            interactive: Return interactive widget if available
            **kwargs: Passed to underlying visualizer
        """
        try:
            if style == VisualizationType.CIRCUIT:
                if not isinstance(data, QuantumCircuit):
                    raise TypeError("Circuit visualization requires QuantumCircuit input")
                fig = circuit_drawer(data, output='mpl', **kwargs)
            else:
                visualizer = self._visualizers[style]
                if not isinstance(data, (Statevector, dict)):
                    raise TypeError(f"{style.name} requires Statevector or counts dict")
                fig = visualizer(data, **kwargs)

            if save_path:
                save_path = Path(save_path)
                fmt = save_path.suffix[1:].lower()
                if fmt not in ['png', 'svg', 'jpg', 'jpeg']:
                    raise ValueError("Unsupported file format")
                fig.savefig(save_path, bbox_inches='tight', format=fmt)
                logger.info(f"Visualization saved to {save_path}")

            return fig._repr_html_() if interactive and hasattr(fig, '_repr_html_') else fig
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            raise

    # === Runtime Backend Management ===
    def _init_runtime_session(self, backend_name: str) -> str:
        """Initialize or reuse runtime session"""
        if not self._runtime_service:
            try:
                self._runtime_service = QiskitRuntimeService()
                logger.info("Runtime service initialized")
            except Exception as e:
                logger.error(f"Runtime init failed: {str(e)}")
                raise

        # Reuse existing session if available
        if backend_name in self._active_sessions:
            return backend_name

        try:
            backend = self._runtime_service.backend(backend_name)
            session = Session(service=self._runtime_service, backend=backend)
            self._active_sessions[backend_name] = session
            logger.info(f"Created new session for {backend_name}")
            return backend_name
        except Exception as e:
            logger.error(f"Session creation failed: {str(e)}")
            raise

    def _get_runtime_sampler(self, backend_name: str, options: Dict) -> Sampler:
        """Get configured sampler for runtime execution"""
        try:
            session = self._active_sessions[backend_name]
            runtime_options = Options()
            for k, v in options.items():
                setattr(runtime_options, k, v)
            return Sampler(session=session, options=runtime_options)
        except Exception as e:
            logger.error(f"Sampler creation failed: {str(e)}")
            raise

    # === Calibration Matrix Validation ===
    def _validate_calibration_matrix(self, matrix: np.ndarray, num_qubits: int):
        """Strict validation of calibration matrix"""
        expected_shape = (2**num_qubits, 2**num_qubits)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"Calibration matrix must be shape {expected_shape} "
                f"for {num_qubits} qubits, got {matrix.shape}"
            )
        if not np.allclose(matrix.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Calibration matrix rows must sum to 1")
        if (matrix < 0).any() or (matrix > 1).any():
            raise ValueError("Calibration matrix values must be in [0,1]")

    def add_calibration_matrix(self, name: str, matrix: np.ndarray, num_qubits: int):
        """Add calibration matrix with strict validation"""
        try:
            self._validate_calibration_matrix(matrix, num_qubits)
            self._calibration_matrices[name] = matrix
            logger.info(f"Added calibration matrix '{name}' for {num_qubits} qubits")
        except Exception as e:
            logger.error(f"Calibration matrix validation failed: {str(e)}")
            raise

    # === Robust Simulation ===
    def simulate(self,
                circuits: Union[QuantumCircuit, List[QuantumCircuit]],
                shots: int = 1024,
                noise_model: Optional[Union[str, noise.NoiseModel]] = None,
                optimize_level: int = 1,
                parameter_bindings: Optional[List[Dict]] = None,
                **kwargs) -> Dict:
        """
        Full-featured simulation with complete error handling
        Args:
            circuits: Circuit(s) to simulate
            shots: Execution shots
            noise_model: Noise model name or object
            optimize_level: Transpiler optimization (0-3)
            parameter_bindings: Parameter sweeps
            **kwargs: Forwarded to backend execution
        Returns:
            Standardized result dictionary
        """
        try:
            # Validate inputs
            if parameter_bindings and not isinstance(circuits, QuantumCircuit):
                raise ValueError("Parameter sweeps require single circuit input")

            # Handle parameter sweeps
            if parameter_bindings:
                circuits = [self.bind_parameters(circuits, params) for params in parameter_bindings]

            circuits = [circuits] if isinstance(circuits, QuantumCircuit) else circuits
            results = []

            for qc in circuits:
                self._validate_circuit(qc)

                # Transpile with caching
                transpiled = self._get_transpiled_circuit(qc, optimize_level)

                # Execute
                if self.backend_type == BackendType.RUNTIME:
                    backend_name = kwargs.pop('backend_name', None) or self._select_optimal_backend(qc.num_qubits)
                    session_id = self._init_runtime_session(backend_name)
                    sampler = self._get_runtime_sampler(session_id, kwargs)
                    result = self._run_on_hardware(transpiled, shots, sampler)
                else:
                    if isinstance(noise_model, str):
                        noise_model = self._get_cached_noise_model(noise_model)
                    result = self._run_locally(transpiled, shots, noise_model, **kwargs)

                results.append(result)

            return self._format_results(results, parameter_bindings is not None)
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            raise

# === Example Usage ===
if __name__ == "__main__":
    # Initialize with all features
    sim = QuantumSimulator(unitary_display_limit=3)
    
    try:
        # Create custom unitary gate
        t_phase = np.array([[1, 0], [0, np.exp(1j * np.pi/4)]])
        sim.add_custom_gate("T_phase", t_phase, [0])
        
        # Create parameterized circuit
        theta = sim.create_parameter("θ")
        qc = QuantumCircuit(2)
        qc.rx(theta, 0)
        qc.append(sim._custom_gates["T_phase"][0], [1])
        
        # Run parameter sweep
        results = sim.simulate(
            qc,
            parameter_bindings=[{"θ": 0}, {"θ": np.pi/2}],
            backend_type=BackendType.STATEVECTOR
        )
        
        # Visualize circuit
        sim.visualize(qc, style=VisualizationType.CIRCUIT, save_path="circuit.png")
        
    finally:
        sim.shutdown()
