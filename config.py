# config.py

import re
from typing import Dict, List, Pattern, Tuple

# Generic numeric fragment: plain decimals, python-style scientific notation ("1.2e-3"),
# and "coefficient x 10^exponent" scientific notation ("1.2x10^-3", "1.2×10−3"), including
# the unicode minus sign (U+2212) commonly produced by PDF text extraction.
# The lookbehind stops a match starting inside a longer number or right after "±" (so "216±31 meV"
# yields 216, not the uncertainty 31); an optional trailing "±err" is consumed and ignored.
NUM = (r'(?<![\d.±])([-−]?\d+(?:\.\d+)?(?:\s*(?:[eE][-−+]?\d+|[xX×\*]\s*10\^?[-−]?\d+))?)'
       r'(?:\s*(?:±|\+/-|\+-)\s*\d+(?:\.\d+)?)?')

# Bounded, non-greedy "skip up to N characters" used between a keyword and its value. Digits
# are deliberately NOT excluded — a chemical formula routinely sits between the keyword and the
# value ("bulk modulus of Li7La3Zr2O12 was found to be 150 GPa") and formulas contain digits.
# Non-greedy backtracking still prefers the closest number+unit match, since every shorter
# candidate gap is tried first and only advances when the unit fails to match right after it.
GAP = r'.{0,80}?'


# Activation-energy (Ea) types, checked per keyword hit by proximity to the extracted value.
# Order only breaks ties. "unspecified" is used when the text names none of them.
EA_PROPERTIES = ("Activation_Energy", "Migration_Barrier")
EA_TYPE_PATTERNS: List[Tuple[str, str]] = [
    ("grain boundary", r'(?i)grain[\s-]*boundar(?:y|ies)|inter-?granular|\bGBs?\b|\bE\s*_?\s*a\s*[,_(\[]\s*gb\b'),
    ("surface", r'(?i)\bsurfaces?\b|\bsurfacic\b|\bE\s*_?\s*a\s*[,_(\[]\s*(?:s|surf|surface)\b'),
    ("bulk", r'(?i)\bbulk\b|\bintra-?granular|\bgrain\s+interior|\bE\s*_?\s*a\s*[,_(\[]\s*(?:b|bulk|g|grain)\b'),
    ("total", r'(?i)\btotal\b|\boverall\b|\bE\s*_?\s*a\s*[,_(\[]\s*(?:t|tot|total)\b'),
]


class SchemaConfig:
    # Scientific NLP Cues
    POSITIVE_CUES = ["our", "this work", "calculated", "measured", "achieved", "exhibits", "shows",
                      "observed", "obtained", "found", "demonstrate", "confirmed"]
    NEGATIVE_CUES = ["previous", "literature", "reported", "ref", "et al", "according to",
                      "compared to", "in contrast to"]

    # Canonical section name -> synonyms seen in real papers (schema section 12).
    SECTION_SYNONYMS: Dict[str, List[str]] = {
        "Abstract": ["abstract"],
        "Introduction": ["introduction", "background"],
        "Methods": [
            "methods", "method", "experimental", "materials and methods",
            "simulation details", "computational details", "computational methods",
            "theory", "theoretical methods", "experimental section", "experimental details",
        ],
        "Results": ["results", "results and discussion"],
        "Discussion": ["discussion"],
        "Conclusion": ["conclusion", "conclusions", "summary", "concluding remarks"],
        "Supplementary": [
            "supplementary information", "supplementary materials", "supporting information",
            "supplementary", "appendix", "electronic supplementary material", "esi",
        ],
        "Boilerplate": [
            "author contributions", "conflict of interest", "conflicts of interest",
            "competing interests", "acknowledgements", "acknowledgments", "data availability",
            "funding", "declaration of competing interest",
        ],
        "References": ["references", "bibliography"],
    }

    SCHEMA: Dict[str, Dict[str, Tuple[str, str]]] = {
        "ELECTROCHEMICAL": {
            # Both properties describe the same underlying quantity (the energy barrier an ion
            # must overcome to hop/migrate) — papers use a wide range of near-synonymous names
            # for it, so both keyword lists are deliberately generous. exporter.py pools both
            # properties into the dedicated activation_energy.csv/sheet.
            "Activation_Energy": (rf'(?i)(?:(?:apparent\s+)?activation\s+energ(?:y|ies)|'
                                   rf'(?:gibbs\s+)?free\s+energy\s+of\s+activation|activation\s+free\s+energy|'
                                   rf'enthalpy\s+of\s+activation|activation\s+enthalpy|E\s*_?a)'
                                   rf'\b{GAP}{NUM}\s*(eV|meV|kJ/mol|kcal/mol)', 'energy'),
            "Migration_Barrier": (rf'(?i)(?:(?:migration|hopping|diffusion|kinetic|reaction|percolation|'
                                   rf'intercalation|charge[- ]transfer|potential\s+energy)\s+barriers?|'
                                   rf'energy\s+barriers?|migration\s+energ(?:y|ies)|hopping\s+energ(?:y|ies)|'
                                   rf'desolvation\s+energ(?:y|ies))\b{GAP}{NUM}\s*(eV|meV|kJ/mol|kcal/mol)', 'energy'),
            "Diffusion_Coefficient": (rf'(?i)(?:diffusion coefficient|diffusivity|D\s*_?(?:Li|Na|K|Mg|Ca|Zn|Al|H)?)\b{GAP}{NUM}\s*(cm2/?s|cm\^?2\s*/?\s*s|m2/?s|m\^?2\s*/?\s*s)', 'diffusion'),
            "Ionic_Conductivity": (rf'(?i)(?:ionic conductivity|total conductivity|σ\s*(?:ion|total|Li|Na)?)\b{GAP}{NUM}\s*(mS/?cm|μS/?cm|uS/?cm|S/?cm)', 'conductivity'),
            "Electronic_Conductivity": (rf'(?i)electronic conductivity\b{GAP}{NUM}\s*(mS/?cm|μS/?cm|uS/?cm|S/?cm)', 'conductivity'),
            "Redox_Potential": (rf'(?i)redox potential\b{GAP}{NUM}\s*(V|mV)', 'voltage'),
            "Stability_Window": (rf'(?i)(?:electrochemical stability window|ESW)\b{GAP}{NUM}\s*(V|mV)', 'voltage'),
            "Voltage": (rf'(?i)(?:average voltage|operating voltage|voltage plateau|voltage)\b{GAP}{NUM}\s*(V|mV)', 'voltage'),
            "Overpotential": (rf'(?i)overpotential\b{GAP}{NUM}\s*(V|mV)', 'voltage'),
            "Capacity": (rf'(?i)(?:specific capacity|discharge capacity|charge capacity|capacity)\b{GAP}{NUM}\s*(mAh/?g|Ah/?kg|mAh/?cm2)', 'capacity'),
            "Specific_Capacity": (rf'(?i)specific capacity\b{GAP}{NUM}\s*(mAh/?g|Ah/?kg)', 'capacity'),
            "Initial_Capacity": (rf'(?i)initial (?:discharge |charge )?capacity\b{GAP}{NUM}\s*(mAh/?g|Ah/?kg)', 'capacity'),
            "Energy_Density": (rf'(?i)energy density\b{GAP}{NUM}\s*(Wh/?kg|Wh/?L|kWh/?kg)', 'energy_density'),
            "Power_Density": (rf'(?i)power density\b{GAP}{NUM}\s*(W/?kg|kW/?kg)', 'power_density'),
            "Current_Density": (rf'(?i)current density\b{GAP}{NUM}\s*(mA/?cm2|A/?cm2|mA/?g|A/?g)', 'current_density'),
            "Critical_Current_Density": (rf'(?i)critical current density\b{GAP}{NUM}\s*(mA/?cm2|A/?cm2)', 'current_density'),
            "Cycle_Life": (rf'(?i)cycle life\b{GAP}{NUM}\s*(cycles)', 'count'),
            "Capacity_Retention": (rf'(?i)capacity retention\b{GAP}{NUM}\s*(%|percent)', 'percent'),
            "Coulombic_Efficiency": (rf'(?i)coulombic efficiency\b{GAP}{NUM}\s*(%|percent)', 'percent'),
            "Rate_Capability": (rf'(?i)rate capability\b{GAP}{NUM}\s*(C|c-rate)', 'string'),
            "C_Rate": (rf'(?i)\bat\s{NUM}\s*C\b', 'string'),
            "Interfacial_Resistance": (rf'(?i)interfacial resistance\b{GAP}{NUM}\s*(ohm\s*cm2|Ω\s*cm2|Ω\s*⋅?\s*cm2)', 'area_resistance'),
        },
        "THERMODYNAMIC": {
            "Formation_Energy": (rf'(?i)formation energy\b{GAP}{NUM}\s*(eV(?:/atom)?|eV/f\.u\.|kJ/mol)', 'energy'),
            "Energy_Above_Hull": (rf'(?i)energy above (?:the )?hull\b{GAP}{NUM}\s*(eV(?:/atom)?|meV(?:/atom)?)', 'energy'),
            "Reaction_Energy": (rf'(?i)reaction energy\b{GAP}{NUM}\s*(eV|eV/atom|kJ/mol)', 'energy'),
            "Surface_Energy": (rf'(?i)surface energy\b{GAP}{NUM}\s*(J/?m2|eV/Å2|eV/A2)', 'surface_energy'),
            "Vacancy_Formation_Energy": (rf'(?i)vacancy formation energy\b{GAP}{NUM}\s*(eV)', 'energy'),
            "Binding_Energy": (rf'(?i)binding energy\b{GAP}{NUM}\s*(eV|kJ/mol)', 'energy'),
            "Adsorption_Energy": (rf'(?i)adsorption energy\b{GAP}{NUM}\s*(eV|kJ/mol)', 'energy'),
        },
        "MECHANICAL": {
            "Bulk_Modulus": (rf'(?i)bulk modulus\b{GAP}{NUM}\s*(GPa|MPa)', 'pressure'),
            "Young_Modulus": (rf"(?i)young'?s modulus\b{GAP}{NUM}\s*(GPa|MPa)", 'pressure'),
            "Shear_Modulus": (rf'(?i)shear modulus\b{GAP}{NUM}\s*(GPa|MPa)', 'pressure'),
            "Poisson_Ratio": (rf"(?i)poisson'?s? ratio\b{GAP}{NUM}\s*()", 'dimensionless'),
        },
        "STRUCTURAL": {
            "Lattice_Parameter_a": (rf'(?i)(?:lattice parameter|lattice constant)\s*a\b{GAP}{NUM}\s*(Å|nm|pm|Angstrom)', 'length'),
            "Lattice_Parameter_c": (rf'(?i)(?:lattice parameter|lattice constant)\s*c\b{GAP}{NUM}\s*(Å|nm|pm|Angstrom)', 'length'),
            "Cell_Volume": (rf'(?i)(?:cell|unit cell) volume\b{GAP}{NUM}\s*(Å3|Å\^3|nm3)', 'volume'),
            "Density": (rf'(?i)(?:mass |bulk )?density\b{GAP}{NUM}\s*(g/?cm3|kg/?m3|g/?cc)', 'density'),
            "Space_Group": (r'(?i)space group\b[^\w]{0,10}([A-Za-z0-9/\-]{2,10}(?:\s*\(No\.\s*\d+\))?)', 'string'),
            "Crystal_System": (r'(?i)\b(cubic|tetragonal|orthorhombic|hexagonal|trigonal|rhombohedral|monoclinic|triclinic)\b', 'string'),
            "Grain_Size": (rf'(?i)grain size\b{GAP}{NUM}\s*(nm|μm|um|mm)', 'length'),
            "Particle_Size": (rf'(?i)particle size\b{GAP}{NUM}\s*(nm|μm|um|mm)', 'length'),
            "Porosity": (rf'(?i)porosity\b{GAP}{NUM}\s*(%|percent)', 'percent'),
            "Relative_Density": (rf'(?i)relative density\b{GAP}{NUM}\s*(%|percent)', 'percent'),
        },
        "ELECTRONIC": {
            "Band_Gap": (rf'(?i)band\s*gap\b{GAP}{NUM}\s*(eV)', 'energy'),
            "Bader_Charge": (rf'(?i)bader charge\b{GAP}{NUM}\s*(e|electrons)?', 'dimensionless'),
            "DOS_Mentioned": (r'(?i)\b(density of states|DOS)\b', 'string'),
            "PDOS_Mentioned": (r'(?i)\b(projected density of states|PDOS)\b', 'string'),
            "COHP_Mentioned": (r'(?i)\b(crystal orbital hamilton population|COHP)\b', 'string'),
        },
        "THERMAL": {
            "Activation_Temperature": (rf'(?i)activation temperature\b{GAP}{NUM}\s*(K|°C|C)', 'temperature'),
            "Melting_Point": (rf'(?i)melting point\b{GAP}{NUM}\s*(K|°C|C)', 'temperature'),
            "Glass_Transition_Temperature": (rf'(?i)glass transition(?:\s*temperature)?\s*(?:Tg)?\b{GAP}{NUM}\s*(K|°C|C)', 'temperature'),
            "Debye_Temperature": (rf'(?i)Debye temperature\b{GAP}{NUM}\s*(K)', 'temperature'),
            "Thermal_Conductivity": (rf'(?i)thermal conductivity\b{GAP}{NUM}\s*(W/?m/?K|W/?\(m·?K\))', 'thermal_conductivity'),
            "Thermal_Expansion_Coefficient": (rf'(?i)thermal expansion coefficient\b{GAP}{NUM}\s*(ppm/?K|10\^?-6/?K|K-1|K\^?-1)', 'dimensionless'),
        },
        "SPECTROSCOPY": {
            "Raman_Peak": (rf'(?i)Raman(?:\s*shift)?\b{GAP}{NUM}\s*(cm-1|cm\^?-1)', 'wavenumber'),
            "FTIR_Peak": (rf'(?i)FTIR\b{GAP}{NUM}\s*(cm-1|cm\^?-1)', 'wavenumber'),
            "XRD_Peak": (rf'(?i)(?:XRD peak|2\s*θ|2theta)\b{GAP}{NUM}\s*(°|deg|degrees)', 'angle'),
        },
        "DFT": {
            "Software": (r'(?i)\b(VASP|Quantum ESPRESSO|CASTEP|CP2K|ABINIT|Gaussian|CRYSTAL|SIESTA|WIEN2k|GPAW|Materials Studio|FHI-aims|ORCA)\b', 'string'),
            "Version": (r'(?i)\bversion\s*([\d.]+)\b', 'string'),
            "Functional": (r'(?i)\b(PBEsol|PBE0|PBE|SCAN|r2SCAN|HSE06|HSE|LDA|GGA\+U|GGA)\b', 'string'),
            "Hubbard_U": (rf'(?i)(?:Hubbard U|DFT\+U|U\s*value)\b{GAP}{NUM}\s*(eV)', 'energy'),
            "Cutoff_Energy": (rf'(?i)(?:cutoff energy|plane[- ]wave cutoff|ENCUT)\b{GAP}{NUM}\s*(eV|Ry)', 'energy_large'),
            "Ecutwfc": (rf'(?i)ecutwfc\b{GAP}{NUM}\s*(Ry|eV)', 'energy_large'),
            "Ecutrho": (rf'(?i)ecutrho\b{GAP}{NUM}\s*(Ry|eV)', 'energy_large'),
            "Pseudopotential": (r'(?i)\b(PAW|norm-conserving|ultrasoft|projector augmented[- ]wave|USPP|NCPP)\b', 'string'),
            "Kpoint_Mesh": (r'(?i)(?:k-?point mesh|Monkhorst-?Pack|k-?mesh|k-?grid)[^\d]{0,20}?(\d+\s*[×xX]\s*\d+\s*[×xX]\s*\d+)', 'string'),
            "SCF_Convergence": (rf'(?i)SCF convergence\b{GAP}{NUM}\s*(eV|Ry)', 'energy_large'),
            "Relaxation_Type": (r'(?i)\b(VC-relax|variable[- ]cell relaxation|ionic relaxation|full relaxation|geometry optimization)\b', 'string'),
            "NEB": (r'(?i)\b(nudged elastic band|NEB)\b', 'string'),
            "Spin_Polarization": (r'(?i)\b(spin[- ]polariz(?:ed|ation)|collinear spin)\b', 'string'),
            "SOC": (r'(?i)\b(spin-orbit coupling|SOC)\b', 'string'),
            "Dispersion_Correction": (r'(?i)\b(DFT-D3|DFT-D2|Grimme dispersion|van der Waals correction|vdW-DF)\b', 'string'),
            "Hybrid_Functional": (r'(?i)\b(hybrid functional|HSE06|PBE0)\b', 'string'),
            "Meta_GGA": (r'(?i)\b(meta-GGA|SCAN|r2SCAN)\b', 'string'),
            "Supercell_Size": (r'(?i)supercell[^\d]{0,20}?(\d+\s*[×xX]\s*\d+\s*[×xX]\s*\d+)', 'string'),
            "Vacancy_Concentration": (rf'(?i)vacancy concentration\b{GAP}{NUM}\s*(%|percent)', 'percent'),
        },
        "AIMD": {
            "Temperature": (rf'(?i)(?:AIMD|simulation)[^.]{{0,40}}?{NUM}\s*K\b', 'temperature'),
            "Pressure": (rf'(?i)pressure\b{GAP}{NUM}\s*(GPa|MPa|bar|atm)', 'pressure'),
            "Simulation_Time": (rf'(?i)simulation time\b{GAP}{NUM}\s*(ps|ns|fs)', 'time'),
            "Timestep": (rf'(?i)time\s*step\b{GAP}{NUM}\s*(fs|ps)', 'time'),
            "Ensemble": (r'\b(NVT|NPT|NVE)\b', 'string'),
            "Supercell": (r'(?i)supercell[^\d]{0,20}?(\d+\s*[×xX]\s*\d+\s*[×xX]\s*\d+)', 'string'),
            "Trajectory_Length": (rf'(?i)trajector(?:y|ies)\b{GAP}{NUM}\s*(ps|ns|fs)', 'time'),
            "MSD_Mentioned": (r'(?i)\b(mean square(?:d)? displacement|MSD)\b', 'string'),
        },
        "ML": {
            "Algorithm": (r'(?i)\b(Random Forest|Gradient Boosting|XGBoost|LightGBM|CatBoost|SVM|Support Vector Machine|GNN|Graph Neural Network|CGCNN|MEGNet|ALIGNN|Neural Network|Deep Learning|Gaussian Process|Kernel Ridge Regression|Linear Regression)\b', 'string'),
            "Descriptor": (r'(?i)\b(Coulomb matrix|SOAP|symmetry function|Voronoi|Magpie descriptor|structural fingerprint)\b', 'string'),
            "Training_Size": (rf'(?i)(?:training (?:set|data)|trained on)\b{GAP}{NUM}\s*(samples|structures|compounds|data points)?', 'count'),
            "Validation_Size": (rf'(?i)validation (?:set|data)\b{GAP}{NUM}\s*(samples|structures|compounds|data points)?', 'count'),
            "Test_Size": (rf'(?i)test (?:set|data)\b{GAP}{NUM}\s*(samples|structures|compounds|data points)?', 'count'),
            "Cross_Validation": (rf'(?i)(\d+)[- ]fold cross[- ]validation', 'count'),
            "Accuracy": (rf'(?i)accuracy\b{GAP}{NUM}\s*(%|percent)?', 'percent'),
            "Precision": (rf'(?i)precision\b{GAP}{NUM}\s*(%|percent)?', 'percent'),
            "Recall": (rf'(?i)recall\b{GAP}{NUM}\s*(%|percent)?', 'percent'),
            "F1_Score": (rf'(?i)F1[- ]?(?:score)?\b{GAP}{NUM}', 'dimensionless'),
            "MAE": (rf'(?i)MAE\b{GAP}{NUM}', 'dimensionless'),
            "RMSE": (rf'(?i)RMSE\b{GAP}{NUM}', 'dimensionless'),
            "R2": (rf'(?i)R\^?2\b{GAP}{NUM}', 'dimensionless'),
            "ROC_AUC": (rf'(?i)(?:ROC[- ]AUC|AUC)\b{GAP}{NUM}', 'dimensionless'),
            "Explainability_Method": (r'(?i)\b(SHAP|LIME)\b', 'string'),
        },
        "SYNTHESIS": {
            "Method": (r'(?i)\b(solid[- ]state|sol[- ]gel|hydrothermal|solvothermal|ball milling|melt quenching|melt-quench|CVD|PLD|co-precipitation|spray pyrolysis|freeze[- ]drying)\b', 'string'),
            "Temperature": (rf'(?i)(?:calcined|sintered|annealed|synthesized)[^\d]{{0,20}}?at{GAP}{NUM}\s*(°C|K)', 'temperature'),
            "Time": (rf'(?i)(?:calcined|sintered|annealed|held)[^\d]{{0,30}}?for{GAP}{NUM}\s*(h|hours|min|minutes)', 'time'),
            "Atmosphere": (r'(?i)\b(argon|nitrogen|air atmosphere|vacuum|inert atmosphere|ambient air)\b', 'string'),
        },
        "CHARACTERIZATION": {
            "Technique": (r'\b(XRD|SEM|TEM|HRTEM|EDS|EDX|XPS|Raman|FTIR|EIS|CV|GITT|NMR|BET|TGA|DSC|SAXS|PDF analysis)\b', 'string'),
        },
    }

    # Compiled once at import time (schema section 18: compile regexes once).
    COMPILED_SCHEMA: Dict[str, Dict[str, Tuple[Pattern, str]]] = {
        cat: {prop: (re.compile(pattern), norm_type) for prop, (pattern, norm_type) in props.items()}
        for cat, props in SCHEMA.items()
    }

    # Equation classification (schema section 10): name/keyword patterns -> canonical equation type.
    EQUATION_PATTERNS: Dict[str, str] = {
        r'(?i)arrhenius': "Arrhenius",
        r'(?i)nernst-?einstein': "Nernst-Einstein",
        r'(?i)mean square(?:d)? displacement|\bMSD\b': "MSD/Diffusion",
        r'(?i)einstein relation': "Einstein Relation",
        r'(?i)nudged elastic band|\bNEB\b': "NEB",
        r'(?i)butler-?volmer': "Butler-Volmer",
        r'(?i)fick\'?s (?:first|second) law': "Fick's Law",
        r'(?i)vegard\'?s law': "Vegard's Law",
    }

    # Ordered computational/experimental workflow stage keywords (schema section 11).
    WORKFLOW_STAGES: List[Tuple[str, str]] = [
        (r'(?i)\b(ICSD|Materials Project|crystal structure database|structure database)\b', "Structure Retrieval"),
        (r'(?i)\b(structural relaxation|geometry optimization|DFT relaxation|VC-relax)\b', "DFT Relaxation"),
        (r'(?i)\bsupercell\b', "Supercell Construction"),
        (r'(?i)\b(ab initio molecular dynamics|AIMD)\b', "AIMD"),
        (r'(?i)\b(mean square(?:d)? displacement|MSD)\b', "MSD Analysis"),
        (r'(?i)\bdiffusion coefficient\b', "Diffusion Analysis"),
        (r'(?i)\barrhenius\b', "Arrhenius Fitting"),
        (r'(?i)\b(ionic conductivity|conductivity)\b', "Conductivity Extraction"),
        (r'(?i)\b(machine learning|neural network|random forest|regression model)\b', "Machine Learning"),
        (r'(?i)\b(experimental validation|experimentally validated|synthesized and tested)\b', "Experimental Validation"),
    ]
