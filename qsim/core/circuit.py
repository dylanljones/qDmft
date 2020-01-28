# -*- coding: utf-8 -*-
"""
Created on 10 Oct 2019
author: Dylan Jones

project: qsim
version: 1.0
"""
import numpy as np
from scitools import Terminal
from .register import Qubit, Clbit, QuRegister, ClRegister
from .utils import Basis, get_info, binary_histogram, plot_binary_histogram, density_matrix
from .backends import StateVector
from .visuals import CircuitString
from .instruction import Instruction, ParameterMap, Gate, Measurement


class Result:

    def __init__(self, data):
        self.data = data
        self.basis = Basis(data.shape[1])

    @classmethod
    def laod(cls, file):
        """ Load a measurement result from a file

        Parameters
        ----------
        file: file-like or str
            File or filename from which the data is loaded. If file is a string or Path,
            a .npy extension will be appended to the file name if it does not already have one.

        Returns
        -------
        res: Result
        """
        data = np.load(file)
        return cls(data)

    @property
    def isnan(self):
        """ bool: check if data is np.nan"""
        return np.all(np.isnan(self.data))

    @property
    def shape(self):
        """tuple: shape of the data array (n_measurements, n_bits)"""
        return self.data.shape

    @property
    def n_samples(self):
        """int: number of measurments"""
        return self.shape[0]

    @property
    def n_bits(self):
        """int: number of bits"""
        return self.shape[1]

    @property
    def labels(self):
        """list of str: Labels of the basis-states of the measured qubits"""
        return [r"$|$" + str(x) + r"$\rangle$" for x in self.basis.state_labels]

    def __bool__(self):
        return not self.isnan

    def save(self, file):
        """ Save the measurement data to a file

        Parameters
        ----------
        file: file-like or str
            File or filename to which the data is saved. If file is a string or Path,
            a .npy extension will be appended to the file name if it does not already have one.
        """
        np.save(file, self.data)

    def binary(self):
        """ np.ndarray: Converts measurement data to binary representation"""
        return (-self.data + 1) / 2

    def mean(self):
        """ np.ndarray: Computes the mean of the measurement data """
        return np.mean(self.data, axis=0)

    def binary_mean(self):
        """ np.ndarray: Computes the mean of the binary data """
        return (-np.sign(self.mean()) + 1) / 2

    def density_matrix(self):
        _, hist = self.histogram()
        return density_matrix(hist)

    def histogram(self, normalize=True):
        """ Computes the binary histogram of the measurement data

        Parameters
        ----------
        normalize: bool, optional
            Flag if the histogram should be normalized

        Returns
        -------
        bins: np.ndarray
        hist: np.ndarray
        """
        return binary_histogram(self.binary(), normalize)

    def show_histogram(self, show=True, padding=0.2, color=None, alpha=0.9, scale=False,
                       max_line=True, lc="r", lw=1):
        bins, hist = self.histogram()
        labels = [r"$|$" + str(x) + r"$\rangle$" for x in self.basis.state_labels]
        plot = plot_binary_histogram(bins, hist, labels, padding, color, alpha, scale, max_line, lc, lw)
        plot.set_labels("State", "p")
        if show:
            plot.show()
        return plot

    def __str__(self):
        string = f"Measurement Result (samples={self.n_samples}):\n"
        string += f"  Mean:   {self.mean()}\n"
        string += f"  Binary: {self.binary_mean()}"
        return string


class Circuit:

    def __init__(self, qubits, clbits=None):
        self.qureg = QuRegister(qubits)
        if clbits is None:
            clbits = len(self.qubits)
        self.clreg = ClRegister(clbits)
        self.basis = Basis(self.n_qubits)
        self.instructions = list()
        self.pmap = ParameterMap.instance()

        self.state = StateVector(self.qubits, self.basis)

    @property
    def qubits(self):
        """ list of Qubit: Qubits of the circuit """
        return self.qureg.bits

    @property
    def n_qubits(self):
        """ int: Number of Qubits in the circuit """
        return self.qureg.n

    @property
    def clbits(self):
        """ list of Clbit: Clbits of the circuit """
        return self.clreg.bits

    @property
    def n_clbits(self):
        """ int: Number of Clbits in the circuit """
        return self.clreg.n

    @property
    def n_params(self):
        """ int: Number of controllable parameters in the circuit """
        return len(self.pmap.params)

    @property
    def params(self):
        """ list: List of controllable parameters in the circuit """
        return self.pmap.params

    @property
    def args(self):
        """ list: List of arguments in the circuit """
        return self.pmap.args

    @property
    def statevector(self):
        """ (N) np.ndarray: Coefficients of the current state of the circuit"""
        return self.state.amp

    def __getitem__(self, item):
        return self.instructions[item]

    def __iter__(self):
        for inst in self.instructions:
            yield inst

    def _build_string(self, show_args=True, padding=1, maxwidth=None):
        s = CircuitString(len(self.qubits), padding)
        for instructions in self.instructions:
            s.add(instructions, show_arg=show_args)
        return s.build(wmax=maxwidth)

    def __repr__(self):
        return f"Circuit(qubits: {self.qubits}, clbits: {self.clbits})"

    def __str__(self):
        return self._build_string(show_args=False)

    def print(self, show_args=True, padding=1, maxwidth=None):
        print(self._build_string(show_args, padding, maxwidth))

    def show(self):
        pass

    def set_state(self, psi=None):
        """ Set the current state-vector

        Parameters
        ----------
        psi: array_like, optional
            Coefficients of a state. The default is the .math:'|0>' state.
        """
        self.state.set(psi)

    def prepare_state(self, *states):
        """ Prepare the current state of the circuit using single qubit states.

        Parameters
        ----------
        states: array_like of (2) array_like
            Single qubit state-vectors.
        """
        self.state.prepare(*states)

    def save_state(self, file):
        """ Save the current state vector of the circuit to a file.

        Parameters
        ----------
        file: file-like or str
            File or filename to which the data is saved. If file is a string or Path,
            a .npy extension will be appended to the file name if it does not already have one.
        """
        return self.state.save_state(file)

    def load_state(self, file):
        """ Load a state vector from a file.

        Parameters
        ----------
        file: file-like or str
            File or filename from which the data is loaded.
        """
        self.state.load_state(file)

    # =========================================================================

    @staticmethod
    def add_custom_gate(name, item):
        Gate.add_custom_gate(name, item)

    def to_string(self, delim="; "):
        info = [f"qubits={self.n_qubits}", f"clbits={self.n_clbits}"]
        string = "".join([x + delim for x in info])
        lines = [string]
        for inst in self.instructions:
            string = inst.to_string()
            lines.append(string)
        return "\n".join(lines)

    @classmethod
    def from_string(cls, string, delim="; "):
        lines = string.splitlines()
        info = lines.pop(0)
        qbits = int(get_info(info, "qubits", delim))
        cbits = int(get_info(info, "clbits", delim))
        self = cls(qbits, cbits)
        for line in lines:
            inst = Instruction.from_string(line, self.qubits, self.clbits, delim)
            self.add(inst)
        return self

    def save(self, file, delim="; "):
        ext = ".circ"
        if not file.endswith(ext):
            file += ext
        with open(file, "w") as f:
            f.write(self.to_string(delim))
        return file

    @classmethod
    def load(cls, file, delim="; "):
        ext = ".circ"
        if not file.endswith(ext):
            file += ext
        with open(file, "r") as f:
            string = f.read()
        return cls.from_string(string, delim)

    # =========================================================================

    def set_params(self, args):
        """ Set all controllable parameters of the circuit

        Parameters
        ----------
        args: list
            All arguments for the parameters of the circuit
        """
        self.pmap.set(args)

    def set_param(self, idx, arg):
        """ Set a controllable parameter of the circuit

        Parameters
        ----------
        idx: int
            Index of the parameter.
        arg: float or int or complex
            Arguments of the parameter.
        """
        self.pmap[idx] = arg

    def add(self, inst):
        """ Appends a configured Intruction to the instruction list of the Circuit. """
        self.instructions.append(inst)
        return inst

    def add_gate(self, name, qubits, con=None, arg=None, argidx=None, n=1, trigger=1):
        """ Configure and append a new Gate-Instruction to the Circuit.

        Parameters
        ----------
        name: str
            Name of the Gate used in the gate dictionary.
        qubits: Qubit or array_like of Qubit
            Qubits on which the gate acts.
        con: Qubit or array_like of Qubit, optional
            List of Qubits that controll the gate. If none are given the Gate isn't controlled.
        arg: int or complex or float or array_like, optional
            Argument of the Gate.
        argidx: int, otional
            Argument-index of the gate if a other parameter should be used.
        n: int, optional
            Number of Qubits the gate acts on. The default is 1.
        trigger: int, optional
            The trigger value if the gate is controlled. The default is 1.

        Returns
        -------
        gate: Gate
        """
        if qubits is None:
            qubits = self.qubits
        qubits = self.qureg.list(qubits)
        con = self.qureg.list(con)
        gate = Gate(name, qubits, con, arg, argidx, n, trigger)
        self.instructions.append(gate)
        return gate

    def add_measurement(self, qubits, clbits, basis=None):
        """ Configure and append a new Measurement-Instruction to the Circuit.

        Parameters
        ----------
        qubits: Qubit or array_like of Qubit
            Qubits that are measured.
        clbits: Clbit or array_like of Clbit
            Clbits to save measurment in.
        basis: str, optional
            The basis in which is measured. The default is the 'z'- or computational-Basis.

        Returns
        -------
        inst: Instruction
        """
        if qubits is None:
            qubits = range(self.n_qubits)
        if clbits is None:
            clbits = qubits
        qubits = self.qureg.list(qubits)
        clbits = self.clreg.list(clbits)
        m = Measurement("m", qubits, clbits, basis=basis)
        self.instructions.append(m)
        return m

    def i(self, qubit=None):
        """ Add an identity gate to the circuit. """
        return self.add_gate("I", qubit)

    def x(self, qubit=None):
        """ Add a Pauli-X gate to the circuit. """
        return self.add_gate("X", qubit)

    def y(self, qubit=None):
        """ Add a Pauli-Y gate to the circuit. """
        return self.add_gate("Y", qubit)

    def z(self, qubit=None):
        """ Add a Pauli-Z gate to the circuit. """
        return self.add_gate("Z", qubit)

    def h(self, qubit=None):
        """ Add a Hadamard (H) gate to the circuit. """
        return self.add_gate("H", qubit)

    def s(self, qubit=None):
        """ Add a phase (S) gate to the circuit. """
        return self.add_gate("S", qubit)

    def t(self, qubit=None):
        """ Add a (T) gate to the circuit. """
        return self.add_gate("T", qubit)

    def rx(self, qubit, arg=np.pi/2, argidx=None):
        """ Add a Pauli-X rotation-gate to the circuit. """
        return self.add_gate("Rx", qubit, arg=arg, argidx=argidx)

    def ry(self, qubit, arg=np.pi/2, argidx=None):
        """ Add a Pauli-Y rotation-gate to the circuit. """
        return self.add_gate("Ry", qubit, arg=arg, argidx=argidx)

    def rz(self, qubit, arg=np.pi/2, argidx=None):
        """ Add a Pauli-Z rotation-gate to the circuit. """
        return self.add_gate("Rz", qubit, arg=arg, argidx=argidx)

    def cx(self, con, qubit, trigger=1):
        """ Add a controlled Pauli-X gate to the circuit. """
        return self.add_gate("X", qubit, con, trigger=trigger)

    def cy(self, con, qubit, trigger=1):
        """ Add a controlled Pauli-Y gate to the circuit. """
        return self.add_gate("Y", qubit, con, trigger=trigger)

    def cz(self, con, qubit, trigger=1):
        """ Add a controlled Pauli-Z gate to the circuit. """
        return self.add_gate("Z", qubit, con, trigger=trigger)

    def ch(self, con, qubit, trigger=1):
        """ Add a controlled Hadamard (H) gate to the circuit. """
        return self.add_gate("H", qubit, con, trigger=trigger)

    def cs(self, con, qubit, trigger=1):
        """ Add a phase (S) gate to the circuit. """
        return self.add_gate("S", qubit, con, trigger=trigger)

    def ct(self, con, qubit, trigger=1):
        """ Add a (T) gate to the circuit. """
        return self.add_gate("T", qubit, con, trigger=trigger)

    def crx(self, con, qubit, arg=np.pi/2, argidx=None, trigger=1):
        """ Add a controlled Pauli-X rotation-gate to the circuit. """
        return self.add_gate("Rx", qubit, con, arg, argidx, trigger=trigger)

    def cry(self, con, qubit, arg=np.pi/2, argidx=None, trigger=1):
        """ Add a controlled Pauli-Y rotation-gate to the circuit. """
        return self.add_gate("Ry", qubit, con, arg, argidx, trigger=trigger)

    def crz(self, con, qubit, arg=np.pi/2, argidx=None, trigger=1):
        """ Add a controlled Pauli-Z rotation-gate to the circuit. """
        return self.add_gate("Rz", qubit, con, arg, argidx, trigger=trigger)

    def xy(self, qubits, arg=0, argidx=None):
        """ Add a XY-gate to the circuit. """
        if not hasattr(qubits[0], "__len__"):
            qubits = [qubits]
        qubits = [self.qureg.list(pair) for pair in qubits]
        gate = Gate("XY", qubits, arg=arg, argidx=argidx, n=2)
        return self.add(gate)

    def b(self, qubits, arg=0, argidx=None):
        """ Add a B-gate to the circuit. """
        if not hasattr(qubits[0], "__len__"):
            qubits = [qubits]
        qubits = [self.qureg.list(pair) for pair in qubits]
        gate = Gate("B", qubits, arg=arg, argidx=argidx, n=2)
        return self.add(gate)

    def m(self, qubits=None, clbits=None):
        """ Add a measurment in the computational basis to the circuit"""
        self.add_measurement(qubits, clbits, "z")

    def mx(self, qubits=None, clbits=None):
        """ Add a Pauli-X measurement to the circuit"""
        self.add_measurement(qubits, clbits, "x")

    def my(self, qubits=None, clbits=None):
        """ Add a Pauli-Y measurement to the circuit"""
        self.add_measurement(qubits, clbits, "y")

    def mz(self, qubits=None, clbits=None):
        """ Add a Pauli-Z measurement to the circuit"""
        self.add_measurement(qubits, clbits, "z")

    # =========================================================================

    def expectation(self, operator, qubit=None):
        r""" Calculates the expectation value of a given operator.

        See Also
        --------
        qsim.core.utils.expectation

        Parameters
        ----------
        operator: np.ndarray
            The exectation of this operator is caluclated
        qubit: Qubit or int, optional
            Qubit if the operator is a single-qubit operator.

        Returns
        -------
        x: float
        """
        if qubit is not None:
            qubit = self.qureg.list(qubit)[0]
        return self.state.expectation(operator, qubit)

    def measure(self, qubits, basis=None):
        """ Measure the state of multiple qubits in a given eigenbasis.

        See Also
        --------
        qsim.core.backends.Statevector.measure

        Parameters
        ----------
        qubits: array_like of Qubit or Qubit
            The qubits that are measured.
        basis: str, optional
            The basis in which is measured. The default is the 'z'- or computational-Basis.

        Returns
        -------
        result: np.ndarray
            Eigenvalue corresponding to the measured eigenstate.
        """
        qubits = self.qureg.list(qubits)
        return self.state.measure(qubits, basis)

    def measure_x(self, qubits, shadow=False, snapshot=True):
        """ Performs a measurement of a single qubit in the x-basis.

        See Also
        --------
        qsim.core.backends.Statevector.measure_x

        Parameters
        ----------
        qubits: array_like of Qubit or Qubit
            The qubits that are measured.
        shadow: bool, optional
            Flag if state should remain in the pre-measurement state.
            The default is 'False'.
        snapshot: bool, optional
            Flag if snapshot of statevector should be saved before measurment.
            The default is 'True'.

        Returns
        -------
        result: np.ndarray
            Eigenvalue corresponding to the measured eigenstate.
        """
        qubits = self.qureg.list(qubits)
        return self.state.measure_x(qubits, shadow, snapshot)

    def measure_y(self, qubits, shadow=False, snapshot=True):
        """ Performs a measurement of a single qubit in the y-basis.

        See Also
        --------
        qsim.core.backends.Statevector.measure_y

        Parameters
        ----------
        qubits: array_like of Qubit or Qubit
            The qubits that are measured.
        shadow: bool, optional
            Flag if state should remain in the pre-measurement state.
            The default is 'False'.
        snapshot: bool, optional
            Flag if snapshot of statevector should be saved before measurment.
            The default is 'True'.

        Returns
        -------
        result: np.ndarray
            Eigenvalue corresponding to the measured eigenstate.
        """
        qubits = self.qureg.list(qubits)
        return self.state.measure_y(qubits, shadow, snapshot)

    def measure_z(self, qubits, shadow=False, snapshot=True):
        """ Performs a measurement of a single qubit in the z-basis.

        See Also
        --------
        qsim.core.backends.Statevector.measure_z

        Parameters
        ----------
        qubits: array_like of Qubit or Qubit
            The qubits that are measured.
        shadow: bool, optional
            Flag if state should remain in the pre-measurement state.
            The default is 'False'.
        snapshot: bool, optional
            Flag if snapshot of statevector should be saved before measurment.
            The default is 'True'.

        Returns
        -------
        result: np.ndarray
            Eigenvalue corresponding to the measured eigenstate.
        """
        qubits = self.qureg.list(qubits)
        return self.state.measure_z(qubits, shadow, snapshot)

    def run_circuit(self, state=None):
        """ Run the configured circuit once.

        After initializing the state of the circuit each of the instructions is applied to the state.
        If there are measurements configured, the data is saved in an array with the same number
        of elements as there are classical bits in the circuit.

        Parameters
        ----------
        state: array_like, optional
            State used to initialize the circuit. The default is the .math:'|0>' state.

        Returns
        -------
        data: np.ndarray of float or np.nan
        """
        self.set_state(state)
        data = np.full(self.n_clbits, np.nan)
        for inst in self.instructions:
            if isinstance(inst, Gate):
                self.state.apply_gate(inst)
            elif isinstance(inst, Measurement):
                eigvals, eigvecs = inst.eigenbasis()
                values = self.state.measure(inst.qubits, eigvals, eigvecs)
                for idx, x in zip(inst.cl_indices, values):
                    data[idx] = x
        return data

    def run(self, shots=1, state=None, verbose=False):
        """ Run the configured circuit multiple times.

        The circuit is run multiple times to extract state data from the circuit.
        The data is returned in the Result object.

        See Also
        --------
        Circuit.run_shot: One run of the circuit.
        Result: object containing measurement data for easier conversion and statistics.

        Parameters
        ----------
        shots: int, optional
            Number of times the circuit is run.
        state: array_like, optional
            State used to initialize the circuit. The default is the .math:'|0>' state.
        verbose: bool, optional
            Flag for printing progress.

        Returns
        -------
        res: Result
        """
        terminal = Terminal()
        header = "Running experiment"
        if verbose:
            terminal.write(header)
        data = np.zeros((shots, self.n_clbits), dtype="float")
        for i in range(shots):
            data[i] = self.run_circuit(state)
            if verbose:
                terminal.updateln(header + f": {100*(i + 1)/shots:.1f}% ({i+1}/{shots})")
        if verbose:
            terminal.writeln()
        return Result(data)
