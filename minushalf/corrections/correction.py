"""
Implements the algorithm that automates the process
of vasp correction and optimizes the necessary parameters.
"""
import os
import shutil
from subprocess import Popen, PIPE
from scipy.optimize import minimize
from loguru import logger
import pandas as pd
from minushalf.utils.orbital import (OrbitalType)
from minushalf.io.vtotal import Vtotal
from minushalf.io.input_file import InputFile
from minushalf.utils.atomic_potential import AtomicPotential
from minushalf.utils.band_structure import BandStructure
from minushalf.utils.negative_band_gap import find_negative_band_gap
from minushalf.io.correction_interface import Correction
from minushalf.softwares.runner import Runner
from minushalf.softwares.software_abstract_factory import SoftwaresAbstractFactory


class DFTCorrection(Correction):
    """
    An algorithm that realizes corrections for
    VASP software
    """

    def __init__(
        self,
        root_folder: str,
        potential_filename: str,
        potential_folder: str,
        exchange_correlation_type: str,
        max_iterations: int,
        software_factory: SoftwaresAbstractFactory,
        runner: Runner,
        calculation_code: str,
        amplitude: float,
        cut_initial_guess: dict,
        tolerance: float,
        input_files: list,
        indirect: bool,
        corrected_potfiles_folder: str,
        correction_type: str,
        band_projection: pd.DataFrame,
        atoms: list,
        is_conduction: bool,
        correction_indexes: dict,
        divide_character: list,
    ):
        """
        init method for the vasp correction class
            Args:
                root_folder (str): Path to the folder where the  correction will be made for each atom

                atoms (list): Atoms name

                potential_filename (str): Name of the potential file used by each
                                          software that performs ab initio calculations

                band_projection (pd.DataFrame): Shows the contribution of each atom in the CBM or VBM

                runner (Runner): class to execute the program that makes ab initio calculations

                only_conduction (bool): Conduction correction without previous valence correction

                indirect (bool): Realize calculations considering indirect gaps
        """
        self.root_folder = root_folder

        self.atoms = atoms

        self.potential_filename = potential_filename

        self.band_projection = band_projection

        self.potential_folder = potential_folder

        self.exchange_correlation_type = exchange_correlation_type

        self.max_iterations = max_iterations

        self.calculation_code = calculation_code

        self.amplitude = amplitude

        self.cut_initial_guess = cut_initial_guess

        self.tolerance = tolerance

        self.runner = runner

        self.software_factory = software_factory

        self.atom_potential = None

        self.sum_correction_percentual = 100

        self.corrected_potfiles_folder = corrected_potfiles_folder

        self.correction_type = correction_type

        self.is_conduction = is_conduction

        self.correction_indexes = correction_indexes

        self.input_files = input_files

        self.indirect = indirect

        self.divide_character = divide_character

    @property
    def potential_folder(self) -> str:
        """
        Returns:
            Name of the folder that helds all the potential files
            initially not corrected.
        """
        return self._potential_folder

    @potential_folder.setter
    def potential_folder(self, path: str) -> None:
        """
        Verify if the folder exists and contains all files needed

        Args:
            path (str): Path of the folder that helds all the potential files
            initially not corrected.
        """

        for atom in self.atoms:
            filename = "{}.{}".format(self.potential_filename.upper(),
                                      atom.lower())
            abs_path = os.path.join(path, filename)
            if not os.path.exists(abs_path):
                logger.error("Potential folder incomplete")
                raise FileNotFoundError("Potential folder lacks of files.")

        self._potential_folder = path

    def execute(self) -> tuple:
        """
        Execute vasp correction algorithm
        """
        # make (valence|conduction) folder in mkpotfiles
        self.root_folder = os.path.join(self.root_folder, self.correction_type)
        if os.path.exists(self.root_folder):
            shutil.rmtree(self.root_folder)
        os.mkdir(self.root_folder)

        cuts_per_atom_orbital = {}
        self._make_corrected_potential_folder()

        self.sum_correction_percentual = self._get_sum_correction_percentual()

        for symbol, orbitals in self.correction_indexes.items():
            for orbital in orbitals:
                cut = self._find_best_correction(symbol, orbital)
                cuts_per_atom_orbital[(symbol, orbital)] = cut

        gap = self._get_result_gap(is_indirect=self.indirect)
        return (cuts_per_atom_orbital, gap)

    def _get_result_gap(self, is_indirect: bool) -> float:
        """
        Return the gap after the optimization of all potfiles
        """
        calculation_folder = "./.minushalf/calculate_{}_gap".format(
            self.correction_type)
        if os.path.exists(calculation_folder):
            shutil.rmtree(calculation_folder)
        os.mkdir(calculation_folder)

        for file in self.input_files:
            shutil.copyfile(file, os.path.join(calculation_folder, file))

        potfile_path = os.path.join(calculation_folder,
                                    self.potential_filename)
        potential_file = open(potfile_path, "w")
        try:
            for atom in self.atoms:
                atom_potfilename = "{}.{}".format(
                    self.potential_filename.upper(), atom.lower())
                atom_potpath = os.path.join(self.corrected_potfiles_folder,
                                            atom_potfilename)
                with open(atom_potpath) as file:
                    potential_file.write(file.read())
        finally:
            potential_file.close()
        # Run ab initio calculations
        self.runner.run(calculation_folder)

        eigenvalues = self.software_factory.get_eigenvalues(
            base_path=calculation_folder)
        fermi_energy = self.software_factory.get_fermi_energy(
            base_path=calculation_folder)
        atoms_map = self.software_factory.get_atoms_map(
            base_path=calculation_folder)
        num_bands = self.software_factory.get_number_of_bands(
            base_path=calculation_folder)
        band_projection_file = self.software_factory.get_band_projection_class(
            base_path=calculation_folder)

        band_structure = BandStructure(eigenvalues, fermi_energy, atoms_map,
                                       num_bands, band_projection_file)

        gap_report = band_structure.band_gap(is_indirect)
        return gap_report["gap"]

    def _make_corrected_potential_folder(self):
        """
        create the folder that will store
        the potfiles corrected with an cut
        value that returns tha maximum gap
        """
        name = self.corrected_potfiles_folder
        if os.path.exists(name):
            shutil.rmtree(name)
        os.mkdir(name)
        for atom in self.atoms:
            try:
                potential_filename = "{}.{}".format(
                    self.potential_filename.upper(), atom.lower())
                potential_path = os.path.join(self.potential_folder,
                                              potential_filename)
                potential_file = open(potential_path, "r")

                new_potential_path = os.path.join(name, potential_filename)
                new_potential_file = open(new_potential_path, "w")
                new_potential_file.write(potential_file.read())

            finally:
                potential_file.close()
                new_potential_file.close()

    def _get_sum_correction_percentual(self) -> float:
        """
        Sum of the percentuals of orbitals
        that will be corrected.
        """
        total_sum = 0
        for symbol, orbitals in self.correction_indexes.items():
            for orbital in orbitals:
                total_sum += self.band_projection[orbital][symbol]

        if total_sum == 0:
            logger.error(
                "No orbital selected for correction. Check you treshhold")
            raise ValueError(
                "No orbital selected for correction. Check you treshhold")

        return total_sum

    def _find_best_correction(self, symbol: str, orbital: str) -> tuple:
        """
        Correct the potcar of the atom symbol in the
        orbital given. Then, find the best cut to the
        the correciton.
            Args:
                symbol (str): Atom symbol
                orbital (str): Orbital type (s,p,d,f)
            Returns:
                gap_and_cut (tuple): Tuple containing
                the optimum cut and the gap generated
                by the correction.
        """
        folder_name = f"mkpotcar_{symbol.lower()}_{orbital.lower()}"
        path = os.path.join(self.root_folder, folder_name)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        self._generate_atom_potential(path, symbol)

        percentuals = {}
        # Check for bonds with equal atoms
        number_equal_neighbors = self.divide_character[(symbol.capitalize(),
                                                        orbital.lower())]

        value = (100 / (1 + number_equal_neighbors)) * (
            self.band_projection[orbital][symbol] /
            self.sum_correction_percentual)
        logger.info(f"percentual of half eletron is {round(value)}")
        percentuals[orbital] = round(value)

        self._generate_occupation_potential(path, percentuals)

        self.atom_potential = self._get_atom_potential_class(path, symbol)

        cut = self._find_cut(symbol, path, orbital)
        self._write_result_in_potfile(symbol, cut, self.amplitude)
        return cut

    def _write_result_in_potfile(
        self,
        symbol: str,
        cut: float,
        amplitude: float,
    ):
        """
        Write the result of the optimization in the
        corrected potfiles folder.

            Args:
                symbol (str): Atom symbol
                amplitude (float): Scale factor to trimming function
                cut (float): Cut radius to the algorithm

        """
        potential = self.atom_potential.correct_potential(
            cut, amplitude, is_conduction=self.is_conduction)
        lines = self.atom_potential.get_corrected_file_lines(potential)

        new_potential_path = os.path.join(
            self.corrected_potfiles_folder,
            "{}.{}".format(self.potential_filename.upper(), symbol.lower()))

        if os.path.exists(new_potential_path):
            os.remove(new_potential_path)

        with open(new_potential_path, "w") as file:
            file.writelines(lines)

    def _generate_atom_potential(
        self,
        base_path: str,
        symbol: str,
    ) -> None:
        """
        Make a dir with the atoms name,generate
        the input file and run the atomic program

            Args:
                symbol (str): Atom symbol
                base_path (str): Path to mkpotcar{symbol}
        """

        folder_path = os.path.join(base_path, "pseudopotential")
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.mkdir(folder_path)
        input_file = InputFile.minimum_setup(
            symbol,
            self.exchange_correlation_type,
            self.max_iterations,
            self.calculation_code,
        )
        input_file.to_file(os.path.join(folder_path, "INP"))
        process = Popen(['minushalf', 'run-atomic', "--quiet"],
                        stdout=PIPE,
                        stderr=PIPE,
                        cwd=folder_path)

        _, stderr = process.communicate()
        if stderr:

            raise Exception("Call to atomic program failed")

    def _generate_occupation_potential(
        self,
        base_path: str,
        percentuals: dict,
    ) -> None:
        """
        Generate the potential for the occupation
        of a fraction of half electron.

            Args:
                base_path (str): Path to mkpotcar{symbol}
                percentual (dict): Dict where the key is the orbital type
                and the value is the percentual to be corrected
        """
        folder_path = os.path.join(base_path, "pseudopotential")
        if not os.path.exists(folder_path):

            raise FileNotFoundError(
                "Folder for pseudopotential does not exist")
        secondary_quantum_numbers = ",".join(
            [str(OrbitalType[key].value) for key in percentuals.keys()])

        joined_percentual = ",".join(
            [str(value) for value in percentuals.values()])

        process = Popen([
            "minushalf", "occupation", "{}".format(secondary_quantum_numbers),
            "{}".format(joined_percentual), "--quiet"
        ],
            stdout=PIPE,
            stderr=PIPE,
            cwd=folder_path)

        _, stderr = process.communicate()
        if stderr:
            raise Exception("Call to occupation command failed")

    def _get_atom_potential_class(self, base_path: str,
                                  symbol: str) -> AtomicPotential:
        """
        Creates atom_potential class

            Args:
                symbol (str): Atom symbol
                base_path (str): Path to mkpotcar{symbol}
        """
        pseudopotential_folder = os.path.join(base_path, "pseudopotential")
        vtotal_path = os.path.join(pseudopotential_folder, "VTOTAL.ae")
        vtotal_occ_path = os.path.join(pseudopotential_folder, "VTOTAL_OCC")
        vtotal = Vtotal.from_file(vtotal_path)
        vtotal_occ = Vtotal.from_file(vtotal_occ_path)

        potential_filename = "{}.{}".format(self.potential_filename.upper(),
                                            symbol.lower())

        potential_class = self.software_factory.get_potential_class(
            potential_filename, self.corrected_potfiles_folder)

        return AtomicPotential(vtotal, vtotal_occ, potential_class)

    def _find_cut(self, symbol: str, base_path: str, orbital: str) -> tuple:
        """
        Find the cut which gives the maximum gap

            Args:
                symbol (str): Atom symbol
                base_path (str): Path to mkpotcar{symbol}
        """
        if not self.indirect:
            folder = os.path.join(base_path, "find_cut")
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.mkdir(folder)
        else:
            base_path = '.'  # Local Path

        function_args = {
            "base_path": base_path,
            "software_factory": self.software_factory,
            "runner": self.runner,
            "symbol": symbol,
            "default_potential_filename": self.potential_filename,
            "atom_potential": self.atom_potential,
            "potfiles_folder": self.potential_folder,
            "amplitude": self.amplitude,
            "atoms": self.atoms,
            "software_files": self.input_files,
            "is_conduction": self.is_conduction,
            "indirect": self.indirect,
            "orbital": orbital
        }
        cut_initial_guess = self.cut_initial_guess[(symbol.capitalize(),
                                                    orbital.lower())]

        result = minimize(find_negative_band_gap,
                          x0=cut_initial_guess,
                          args=(function_args),
                          method="Nelder-Mead",
                          options={'xatol': self.tolerance})

        if not result.success:
            logger.error("Optimization failed")
            raise Exception("Optimization failed.")

        cut = result.x[0]
        return cut



    """
Implements the algorithm that automates the process
of QE correction and optimizes the necessary parameters.

Workflow differences from DFTCorrection (VASP):
------------------------------------------------
1. Atomic program  : ld1.x instead of ATOM
2. Potential files : one .upf per element instead of concatenated POTCAR
3. Vtotal source   : extracted from ld1.x output UPF instead of VTOTAL.ae
4. Pseudo pipeline : requires virtual_v2.x conversion step before ld1.x
5. Potential assembly : copy individual .upf files, no concatenation
"""



class QECorrection(Correction):
    """
    An algorithm that realizes DFT-1/2 corrections for
    Quantum ESPRESSO software.

    Mirrors DFTCorrection but replaces all ATOM/POTCAR-specific
    steps with their ld1.x/UPF equivalents.
    """

    def __init__(
        self,
        root_folder: str,
        potential_filename: str,       # e.g. "upf" — extension used to find .upf files
        potential_folder: str,         # folder with uncorrected .upf files per element
        exchange_correlation_type: str,
        max_iterations: int,
        software_factory: SoftwaresAbstractFactory,
        runner: Runner,
        calculation_code: str,
        amplitude: float,
        cut_initial_guess: dict,
        tolerance: float,
        input_files: list,             # QE input files (scf.in, etc.)
        indirect: bool,
        corrected_potfiles_folder: str,
        correction_type: str,
        band_projection: pd.DataFrame,
        atoms: list,
        is_conduction: bool,
        correction_indexes: dict,
        divide_character: list,
    ):
        """
        Same parameters as DFTCorrection with two additions:

        potential_filename (str): for QE this is the UPF file extension
            convention used in potential_folder. Files are expected to be
            named {SYMBOL}.upf (e.g. Al.upf, N.upf).
        """
        self.root_folder              = root_folder
        self.atoms                    = atoms
        self.potential_filename       = potential_filename
        self.band_projection          = band_projection
        self.potential_folder         = potential_folder
        self.exchange_correlation_type = exchange_correlation_type
        self.max_iterations           = max_iterations
        self.calculation_code         = calculation_code
        self.amplitude                = amplitude
        self.cut_initial_guess        = cut_initial_guess
        self.tolerance                = tolerance
        self.runner                   = runner
        self.software_factory         = software_factory
        self.atom_potential           = None
        self.sum_correction_percentual = 100
        self.corrected_potfiles_folder = corrected_potfiles_folder
        self.correction_type          = correction_type
        self.is_conduction            = is_conduction
        self.correction_indexes       = correction_indexes
        self.input_files              = input_files
        self.indirect                 = indirect
        self.divide_character         = divide_character

    @property
    def potential_folder(self) -> str:
        """
        Returns:
            Name of the folder that helds all the potential files
            initially not corrected.
        """
        return self._potential_folder


    @potential_folder.setter
    def potential_folder(self, path: str) -> None:
        """
        Verify that the potential file exists inside the potential folder.

        The potential filename is provided by self.potential_filename.
        """
        abs_path = os.path.join(path, self.potential_filename)

        if not os.path.exists(abs_path):
            logger.error("Potential folder incomplete")
            raise FileNotFoundError(
                f"Potential folder lacks {self.potential_filename}."
            )

        self._potential_folder = path
    # ------------------------------------------------------------------ #
    #  Public entry point — mirrors DFTCorrection.execute                  #
    # ------------------------------------------------------------------ #

    def execute(self) -> tuple:
        """
        Execute the Qunatum ESPRESSO correction algorithm.
        
        """
        self.root_folder = os.path.join(self.root_folder, self.correction_type)
        if os.path.exists(self.root_folder):
            shutil.rmtree(self.root_folder)
        os.mkdir(self.root_folder)

        cuts_per_atom_orbital = {}
        self._make_corrected_potential_folder()
        self.sum_correction_percentual = self._get_sum_correction_percentual()

        for symbol, orbitals in self.correction_indexes.items():
            for orbital in orbitals:
                cut = self._find_best_correction(symbol, orbital)
                cuts_per_atom_orbital[(symbol, orbital)] = cut

        gap = self._get_result_gap(is_indirect=self.indirect)
        return (cuts_per_atom_orbital, gap)

    # ------------------------------------------------------------------ #
    #  Gap calculation — mostly software-agnostic via factory              #
    # ------------------------------------------------------------------ #

    def _get_result_gap(self, is_indirect: bool) -> float:
        """
        Run a final ab initio calculation with all corrected UPF files
        and return the band gap.

        Differences from DFTCorrection._get_result_gap:
          - Copies individual .upf files per atom instead of
            concatenating into a single POTCAR.
          - Passes software_input to factory calls so the correct
            prefix/outdir is resolved.
        """
        calculation_folder = f"./.minushalf/calculate_{self.correction_type}_gap"
        if os.path.exists(calculation_folder):
            shutil.rmtree(calculation_folder)
        os.mkdir(calculation_folder)

        # Copy QE input files (scf.in, pseudos, etc.)
        for file in self.input_files:
            shutil.copyfile(file, os.path.join(calculation_folder, file))

        # Copy corrected UPF files — one per atom, no concatenation
        for atom in self.atoms:
            upf_filename = f"{atom}.upf"
            src  = os.path.join(self.corrected_potfiles_folder, upf_filename)
            dest = os.path.join(calculation_folder, upf_filename)
            shutil.copyfile(src, dest)

        self.runner.run(calculation_folder)

        # All factory calls pass input_files to resolve prefix/outdir
        eigenvalues = self.software_factory.get_eigenvalues(
            base_path=calculation_folder,
            software_input=self.input_files)
        fermi_energy = self.software_factory.get_fermi_energy(
            base_path=calculation_folder,
            software_input=self.input_files)
        atoms_map = self.software_factory.get_atoms_map(
            base_path=calculation_folder,
            software_input=self.input_files)
        num_bands = self.software_factory.get_number_of_bands(
            base_path=calculation_folder,
            software_input=self.input_files)
        band_projection_file = self.software_factory.get_band_projection_class(
            base_path=calculation_folder,
            software_input=self.input_files)

        band_structure = BandStructure(eigenvalues, fermi_energy, atoms_map,
                                       num_bands, band_projection_file)
        gap_report = band_structure.band_gap(is_indirect)
        return gap_report["gap"]

    # ------------------------------------------------------------------ #
    #  Potential folder management                                         #
    # ------------------------------------------------------------------ #

    def _make_corrected_potential_folder(self) -> None:
        """
        Create the folder that stores corrected UPF files.
        Copies the original uncorrected .upf for each atom as starting point.
        Mirrors DFTCorrection._make_corrected_potential_folder but for UPF files.
        """
        name = self.corrected_potfiles_folder
        if os.path.exists(name):
            shutil.rmtree(name)
        os.mkdir(name)

        # Loop through input_files, skipping the first item
        for upf in self.input_files[1:]:
            src  = os.path.join(self.potential_folder, upf)
            dest = os.path.join(name, upf)
            try:
                shutil.copyfile(src, dest)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"{e}. Check if the potential_folder is correctly given in the configuration file."
                )
    # ------------------------------------------------------------------ #
    #  Correction loop — mirrors DFTCorrection._find_best_correction       #
    # ------------------------------------------------------------------ #

    def _find_best_correction(self, symbol: str, orbital: str) -> float:
        """
        For a given atom symbol and orbital type:
          1. Create working folder mkpotcar_{symbol}_{orbital}/
          2. Run ld1.x pipeline to generate atomic potential
          3. Run occupation command
          4. Find the optimal cut via Nelder-Mead
          5. Write the corrected UPF to corrected_potfiles_folder

        Mirrors DFTCorrection._find_best_correction exactly in structure.
        """
        folder_name = f"mkpotcar_{symbol.lower()}_{orbital.lower()}"
        path = os.path.join(self.root_folder, folder_name)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        self._generate_atom_potential(path, symbol)

        percentuals = {}
        number_equal_neighbors = self.divide_character[(symbol.capitalize(),
                                                        orbital.lower())]
        value = (100 / (1 + number_equal_neighbors)) * (
            self.band_projection[orbital][symbol] /
            self.sum_correction_percentual)
        logger.info(f"percentual of half electron is {round(value)}")
        percentuals[orbital] = round(value)

        self._generate_occupation_potential(path, percentuals)

        self.atom_potential = self._get_atom_potential_class(path, symbol)

        cut = self._find_cut(symbol, path, orbital)
        self._write_result_in_potfile(symbol, cut, self.amplitude)
        return cut

    # ------------------------------------------------------------------ #
    #  Atomic program pipeline — QE-specific                               #
    # ------------------------------------------------------------------ #

    def _generate_atom_potential(self, base_path: str, symbol: str) -> None:
        """
        QE equivalent of DFTCorrection._generate_atom_potential.

        Steps (mirrors the bash script workflow):
          1. Create pseudopotential/ subfolder
          2. Run virtual_v2.x to convert the original UPF to a
             zero-starting radial grid → produces NewPseudo.UPF
          3. Write the ld1.x input file (INP.ldx) using InputFile
             with software="QE", including &input and &test namelists
          4. Run ld1.x → produces {symbol}-05.upf.tmp
          5. Run virtual_v2.x again to convert back to original grid
             → produces the corrected {symbol}-05.upf

        Unlike VASP, does NOT call `minushalf run-atomic` — the ld1.x
        pipeline is self-contained here because it requires multiple
        steps and intermediate files.
        """
        folder_path = os.path.join(base_path, "pseudopotential")
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.mkdir(folder_path)

        # Step 1: convert original UPF to zero-starting grid
        original_upf = os.path.join(self.potential_folder, f"{symbol}.upf")
        self._run_virtual_v2(
            pseudo_1=original_upf,
            pseudo_2=original_upf,
            mix=0,
            cwd=folder_path
        )
        # virtual_v2.x always outputs NewPseudo.UPF
        converted_pseudo = os.path.join(folder_path, "NewPseudo.UPF")

        # Step 2: write ld1.x input file
        # TODO: cut value comes from cut_initial_guess at this stage;
        # the actual optimization loop will call _generate_atom_potential
        # repeatedly via _find_cut → find_negative_band_gap.
        # Pass a placeholder cut here; the real cut is set in _find_cut.
        input_file = InputFile.minimum_setup(
            software="QE",
            cut=self.cut_initial_guess.get((symbol.capitalize(), "s"), 2.0),
            chemical_symbol=symbol,
            exchange_correlation_code=self.exchange_correlation_type,
            maximum_iterations=self.max_iterations,
            calculation_code=self.calculation_code,
        )
        input_file.to_file(os.path.join(folder_path, "INP.ldx"))

        # Step 3: run ld1.x
        self._run_ld1(folder_path)

        # Step 4: convert ld1.x output back to original grid
        lda05_tmp = os.path.join(folder_path, f"{symbol}-05.upf.tmp")
        self._run_virtual_v2(
            pseudo_1=original_upf,
            pseudo_2=lda05_tmp,
            mix=0,
            cwd=folder_path
        )
        # Rename NewPseudo.UPF to {symbol}-05.upf
        shutil.move(
            os.path.join(folder_path, "NewPseudo.UPF"),
            os.path.join(folder_path, f"{symbol}-05.upf")
        )

    def _run_virtual_v2(self, pseudo_1: str, pseudo_2: str,
                        mix: float, cwd: str) -> None:
        """
        Run virtual_v2.x with the given pseudopotential inputs.

        virtual_v2.x reads from stdin:
            pseudo_1
            pseudo_2
            mix    (x: produces x*pseudo_1 + (1-x)*pseudo_2)

        Always outputs NewPseudo.UPF in the working directory.

        Args:
            pseudo_1 (str): path to first pseudopotential
            pseudo_2 (str): path to second pseudopotential
            mix      (float): mixing factor x
            cwd      (str): working directory for the process
        """
        # TODO: runner should provide the virtual_v2.x executable path
        # For now using a direct Popen call as placeholder
        stdin_content = f"{pseudo_1}\n{pseudo_2}\n{mix}\n"
        process = Popen(
            ["virtual_v2.x"],
            stdin=PIPE, stdout=PIPE, stderr=PIPE,
            cwd=cwd
        )
        _, stderr = process.communicate(input=stdin_content.encode())
        if stderr:
            raise Exception(f"virtual_v2.x failed: {stderr.decode()}")

    def _run_ld1(self, cwd: str) -> None:
        """
        Run ld1.x reading from INP.ldx in the given directory.
        Produces {symbol}-05.upf.tmp as output.

        Args:
            cwd (str): working directory containing INP.ldx
        """
        # TODO: runner should provide the ld1.x executable path
        with open(os.path.join(cwd, "INP.ldx")) as inp:
            process = Popen(
                ["ld1.x"],
                stdin=inp, stdout=PIPE, stderr=PIPE,
                cwd=cwd
            )
        _, stderr = process.communicate()
        if stderr:
            raise Exception(f"ld1.x failed: {stderr.decode()}")

    def _generate_occupation_potential(self, base_path: str,
                                       percentuals: dict) -> None:
        """
        Generate the potential for the fractional occupation.

        Identical to DFTCorrection._generate_occupation_potential:
        calls `minushalf occupation` which modifies the INP file
        and re-runs the atomic program.

        NOTE: for QE this needs to re-run ld1.x instead of ATOM.
        This is handled by `minushalf run-atomic` detecting the software
        from the INP file extension (INP.ae → ATOM, INP.ldx → ld1.x).
        TODO: verify run-atomic dispatches correctly for QE.
        """
        folder_path = os.path.join(base_path, "pseudopotential")
        secondary_quantum_numbers = ",".join(
            [str(OrbitalType[key].value) for key in percentuals.keys()])
        joined_percentual = ",".join(
            [str(value) for value in percentuals.values()])

        process = Popen([
            "minushalf", "occupation",
            secondary_quantum_numbers,
            joined_percentual,
            "--quiet"
        ],
            stdout=PIPE, stderr=PIPE,
            cwd=folder_path
        )
        _, stderr = process.communicate()
        if stderr:
            raise Exception("Call to occupation command failed")

    # ------------------------------------------------------------------ #
    #  Potential class — QE-specific                                       #
    # ------------------------------------------------------------------ #

    def _get_atom_potential_class(self, base_path: str,
                                  symbol: str) -> AtomicPotential:
        """
        QE equivalent of DFTCorrection._get_atom_potential_class.

        Key difference: VASP reads VTOTAL.ae and VTOTAL_OCC from ATOM output.
        For QE, the potential data is extracted from the UPF file produced
        by ld1.x — specifically the PP_LOCAL block of {symbol}-05.upf.

        TODO: determine whether Vtotal can be extracted from the ld1.x
        output UPF directly (PP_LOCAL block) or whether ld1.x produces
        a separate potential output file that can be parsed instead.
        This is the most uncertain step in the QE port and may require
        a new Vtotal-equivalent parser for ld1.x output.
        """
        pseudopotential_folder = os.path.join(base_path, "pseudopotential")

        # TODO: replace with QE-specific Vtotal extraction
        vtotal_path     = os.path.join(pseudopotential_folder, "VTOTAL.ae")
        vtotal_occ_path = os.path.join(pseudopotential_folder, "VTOTAL_OCC")
        vtotal     = None  # placeholder
        vtotal_occ = None  # placeholder

        upf_filename  = f"{symbol}.upf"
        potential_class = self.software_factory.get_potential_class(
            upf_filename, self.corrected_potfiles_folder)

        return AtomicPotential(vtotal, vtotal_occ, potential_class)

    # ------------------------------------------------------------------ #
    #  Cut optimization — identical to DFTCorrection                       #
    # ------------------------------------------------------------------ #

    def _find_cut(self, symbol: str, base_path: str, orbital: str) -> float:
        """
        Find the cut which gives the maximum gap using Nelder-Mead.

        Identical to DFTCorrection._find_cut — the optimization loop
        itself is software-agnostic. Software-specific behaviour is
        encapsulated in find_negative_band_gap via the factory and runner.
        """
        if not self.indirect:
            folder = os.path.join(base_path, "find_cut")
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.mkdir(folder)
        else:
            base_path = '.'

        function_args = {
            "base_path":                  base_path,
            "software_factory":           self.software_factory,
            "runner":                     self.runner,
            "symbol":                     symbol,
            "default_potential_filename": self.potential_filename,
            "atom_potential":             self.atom_potential,
            "potfiles_folder":            self.potential_folder,
            "amplitude":                  self.amplitude,
            "atoms":                      self.atoms,
            "software_files":             self.input_files,
            "is_conduction":              self.is_conduction,
            "indirect":                   self.indirect,
            "orbital":                    orbital,
        }
        cut_initial_guess = self.cut_initial_guess[(symbol.capitalize(),
                                                    orbital.lower())]
        result = minimize(
            find_negative_band_gap,
            x0=cut_initial_guess,
            args=(function_args),
            method="Nelder-Mead",
            options={'xatol': self.tolerance}
        )
        if not result.success:
            logger.error("Optimization failed")
            raise Exception("Optimization failed.")

        return result.x[0]

    def _write_result_in_potfile(self, symbol: str, cut: float,
                                 amplitude: float) -> None:
        """
        Write the corrected UPF to the corrected potfiles folder.

        Mirrors DFTCorrection._write_result_in_potfile but targets
        {symbol}.upf instead of POTCAR.{symbol}.
        """
        potential = self.atom_potential.correct_potential(
            cut, amplitude, is_conduction=self.is_conduction)
        lines = self.atom_potential.get_corrected_file_lines(potential)

        new_potential_path = os.path.join(
            self.corrected_potfiles_folder, f"{symbol}.upf")

        if os.path.exists(new_potential_path):
            os.remove(new_potential_path)

        with open(new_potential_path, "w") as file:
            file.writelines(lines)

    # ------------------------------------------------------------------ #
    #  Helpers — identical to DFTCorrection                                #
    # ------------------------------------------------------------------ #

    def _get_sum_correction_percentual(self) -> float:
        """
        Sum of the percentuals of orbitals to be corrected.
        Identical to DFTCorrection._get_sum_correction_percentual.
        """
        total_sum = 0
        for symbol, orbitals in self.correction_indexes.items():
            for orbital in orbitals:
                total_sum += self.band_projection[orbital][symbol]

        if total_sum == 0:
            logger.error(
                "No orbital selected for correction. Check your threshold.")
            raise ValueError(
                "No orbital selected for correction. Check your threshold.")

        return total_sum


"""
VASP Example yaml:
software: VASP

vasp:
    command: ['mpirun', '-np', '4', 'vasp']

atomic_program:
    exchange_correlation_code: pb
    calculation_code: ae
    max_iterations: 100

correction:
    correction_code: vc
    potfiles_folder: ./minushalf_potfiles
    amplitude: 1.0
    valence_cut_guess: [["C", "p", 3.2]]
    conduction_cut_guess: [["Si", "p", 3.0]]
    tolerance: 0.01
    fractional_valence_treshold: 10
    fractional_conduction_treshold: 10
    overwrite_vbm: [4, 9]
    overwrite_cbm: [1, 3]
    inplace: False
    divide_character: [["C", "p", 1]]
    vbm_characters: [["C", "s", 34]]
    cbm_characters: [["C", "s", 50]]

    


QE example yaml:
software: QE

qe:
    pw_command: ['mpirun', '-np' , '4', 'pw.x']
    projwfc_command: ['mpirun', '-np' , '4', 'projwfc.x']
    ld1_command: ['mpirun', '-np' , '4', 'ld1.x']
    virtual_v2_command: ['mpirun', '-np' , '4', 'virtual_v2.x']

    scf_input: ./scf.in

atomic_program:
    exchange_correlation_code: pb
    calculation_code: ae
    max_iterations: 100

correction:
    correction_code: vc
    potfiles_folder: ./minushalf_potfiles
    amplitude: 1.0
    valence_cut_guess: [["C", "p", 3.2]]
    conduction_cut_guess: [["Si", "p", 3.0]]
    tolerance: 0.01
    fractional_valence_treshold: 10
    fractional_conduction_treshold: 10
    overwrite_vbm: [4, 9]
    overwrite_cbm: [1, 3]
    inplace: False
    divide_character: [["C", "p", 1]]
    vbm_characters: [["C", "s", 34]]
    cbm_characters: [["C", "s", 50]]

"""