# 🧠 QuantumSimulator

A production-grade Python-based quantum simulator framework built on Qiskit.  
It includes advanced features for custom gate handling, parameter expression tracking, calibration validation, local and runtime execution, and comprehensive visualization.

---

## 🚀 Features

- ✅ Supports **custom unitary gates** with validation
- ✅ Built-in **session save/load** using QPY + JSON (incl. parameter expressions)
- ✅ Supports **parameter sweeps** and symbolic circuit creation
- ✅ Full **backend flexibility**:
  - `STATEVECTOR`, `QASM`, `UNITARY`, `GPU`, `RUNTIME`
- ✅ IBM **Runtime session management** with sampler reuse
- ✅ Strict **calibration matrix validation** and loading
- ✅ Interactive and static **visualizations**:
  - Bloch Sphere
  - City plot
  - QSphere
  - Pauli vector
  - Histogram
  - Circuit diagram
- ✅ Structured logging (`quantum_simulator.log`)
- ✅ Multithreaded execution with `ThreadPoolExecutor`

---

## 📦 Dependencies

- `qiskit`
- `matplotlib`
- `numpy`
- `concurrent.futures`
- `logging`
- `asyncio`
- `qiskit_ibm_runtime`
- `qiskit.providers.fake_provider`

> Install dependencies:
```bash
pip install qiskit qiskit-ibm-runtime matplotlib
