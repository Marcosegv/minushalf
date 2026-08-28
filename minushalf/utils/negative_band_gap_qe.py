"""
Returns band-gap with the sinal changed, so
one can use minimization algorithms to find the
cut value that results in the maximum band_gap
"""
import os
import shutil
import math
from loguru import logger
from minushalf.softwares.software_abstract_factory import SoftwaresAbstractFactory
from minushalf.utils.band_structure import BandStructure
from minushalf.utils.software_output import get_output_filenames


def _set_up_cut_folder(base_path: str, input_files: list, cut: float) -> str:
    """
        Creates and populates the folder where the first principles calculations will be done

        Args:
            base_path (str): Path where the folder will be created
            cut (float): Distance to trimm the potential
            input_files (List[str]): List of input files

        Returns:
            cut_folder (str): Path to folder where the first principles calculations will be done
    """
    cut_folder = _create_cut_folder(base_path, cut)
    _copy_input_files(input_files, cut_folder)
    return cut_folder


def _create_cut_folder(base_path: str, cut: float) -> str:
    """
        Creates the folder where the first principles calculations will be done

        Args:

            base_path (str): Path where the folder will be created
            cut (float): Distance to trimm the potential

        Returns:
            cut_folder (str): Path to folder where the first principles calculations will be done
    """
    find_cut_path = os.path.join(base_path, "find_cut")
    cut_folder = os.path.join(find_cut_path, "cut_{:.2f}".format(cut))

    if os.path.exists(cut_folder):
        shutil.rmtree(cut_folder)
    os.mkdir(cut_folder)

    return cut_folder


def _copy_input_files(input_files: list, destination_folder: str) -> None:
    """
        Copy the input files for the first principles calculations

        Args:
            input_files (List[str]): List of input files
            destination_folder (str): Path to which files will be copied
    """

    for file in input_files:
        shutil.copyfile(file, os.path.join(destination_folder, file))


def _get_gap(software_factory: SoftwaresAbstractFactory,
             cut_folder: str, is_indirect: bool, software_files: list) -> float:
    """
        Returns the gap value

        Args:
            software_factory (SoftwaresAbstractFactory) : Get informations for output files of the first principle calculations
            cut_folder (str): Folder where first principles calculations were made
            is_indirect (bool): This parameter determines whether a band gap calculation should be performed across various k-points.

        Returns:
            gap (float): Gap of the semiconductor material
    """

    filenames = get_output_filenames('QE', software_files[0])

    eigenvalues          = software_factory.get_eigenvalues(
                               filename=filenames["eigenvalues"],
                               base_path=cut_folder)
    fermi_energy         = software_factory.get_fermi_energy(
                               filename=filenames["fermi_energy"],
                               base_path=cut_folder)
    atoms_map            = software_factory.get_atoms_map(
                               filename=filenames["atoms_map"],
                               base_path=cut_folder)
    num_bands            = software_factory.get_number_of_bands(
                               filename=filenames["number_of_bands"],
                               base_path=cut_folder)
    band_projection_file = software_factory.get_band_projection_class(
                               filename=filenames["band_projection"],
                               base_path=cut_folder)

    band_structure = BandStructure(eigenvalues, fermi_energy, atoms_map,
                                   num_bands, band_projection_file)

    gap_report = band_structure.band_gap(is_indirect=is_indirect)
    return gap_report["gap"]


def find_negative_band_gap_qe(cuts: list, *args: tuple) -> float:
    """
                Run vasp and return the gap value multiplied by -1

                Args:
                    cuts (float): List of cuts

                    args (tuple): tuple containning a dictionary with the fields
                                   base_path (str): Path to mkpotcar{symbol}_{orbital}
                                   symbol (str): Atom symbol
                                   default_potential_filename (str): The default potential filename for each software
                                   potfiles_folder (str): Folder containing unmodified potfiles
                                   amplitude (float): scale factor to trimming function
                                   runner (Runner): runner for the software
                                   software_factory(SoftwaresAbstractFactory): Factory for each software
                                   software_files (list): Aditional files besides potential file to make ab initio calculations

                Returns:

                    negative_gap (float): band gap multiplied for -1
    """
    extra_args = args[0]
    cut = cuts[0]
    runner = extra_args["runner"]
    software_factory = extra_args["software_factory"]
    indirect = extra_args["indirect"]
    is_conduction = extra_args["is_conduction"]

    # Add condition to include indirect calculations
    if not indirect:
        cut_folder = _set_up_cut_folder(extra_args["base_path"],
                                        extra_args["software_files"], cut)
    else:
        cut_folder = '.'

    runner.run(cut_folder)

    is_indirect = extra_args["indirect"]
    gap = _get_gap(software_factory, cut_folder, is_indirect, extra_args["software_files"])

    # Logger
    if is_conduction:
        logger.info("CONDUCTION CORRECTION: Element {} - Orbital {}".format(
            extra_args["symbol"], extra_args["orbital"]))
        logger.info(
            "CONDUCTION CORRECTION: Current CUT value is {:.2f} a.u".format(
                cut))
        logger.info(
            "CONDUCTION CORRECTION: Current Gap value is {:.2f} eV".format(
                gap))
    else:
        logger.info("VALENCE CORRECTION: Element {} - Orbital {}".format(
            extra_args["symbol"], extra_args["orbital"]))
        logger.info(
            "VALENCE CORRECTION: Current CUT value is {:.2f} a.u".format(cut))
        logger.info(
            "VALENCE CORRECTION: Current Gap value is {:.2f} eV".format(gap))

    if cut < 0.5:
        logger.warning(
            "Pay attention, CUT values less than 0.5 a.u might have no physical meaning."
        )

    return (-1) * gap
