from mealpy import FloatVar

from enoppy.paper_based.rwco_2020 import (
    PressureVesselDesignProblem,
    WeldedBeamDesignProblem,
    ThreeBarTrussDesignProblem,
    MultipleDiskClutchBrakeDesignProblem,
)

from enoppy.paper_based.pdo_2022 import (
    TubularColumnProblem,
    CorrugatedBulkheadProblem,
)


def engineering_problems():
    EP1 = PressureVesselDesignProblem()
    P1 = {
        "bounds": FloatVar(lb=EP1.lb, ub=EP1.ub),
        "minmax": "min",
        "fit_func": EP1.evaluate,
        "name": "Pressure Vessel",
        "log_to": None,
        "save_population": False,
    }

    EP2 = WeldedBeamDesignProblem()
    P2 = {
        "bounds": FloatVar(lb=EP2.lb, ub=EP2.ub),
        "minmax": "min",
        "fit_func": EP2.evaluate,
        "name": "Welded Beam",
        "log_to": None,
        "save_population": False,
    }

    EP3 = ThreeBarTrussDesignProblem()
    P3 = {
        "bounds": FloatVar(lb=EP3.lb, ub=EP3.ub),
        "minmax": "min",
        "fit_func": EP3.evaluate,
        "name": "Three Bar Truss",
        "log_to": None,
        "save_population": False,
    }

    EP4 = MultipleDiskClutchBrakeDesignProblem()
    P4 = {
        "bounds": FloatVar(lb=EP4.lb, ub=EP4.ub),
        "minmax": "min",
        "fit_func": EP4.evaluate,
        "name": "Multiple Disk",
        "log_to": None,
        "save_population": False,
    }

    EP5 = TubularColumnProblem()
    P5 = {
        "bounds": FloatVar(lb=EP5.lb, ub=EP5.ub),
        "minmax": "min",
        "fit_func": EP5.evaluate,
        "name": "Tubular-Column",
        "log_to": None,
        "save_population": False,
    }

    EP6 = CorrugatedBulkheadProblem()
    P6 = {
        "bounds": FloatVar(lb=EP6.lb, ub=EP6.ub),
        "minmax": "min",
        "fit_func": EP6.evaluate,
        "name": "Corrugated-Bulkhead",
        "log_to": None,
        "save_population": False,
    }

    names = [
        "Pressure Vessel",
        "Welded Beam",
        "Three Bar Truss",
        "Multiple Disk",
        "Tubular-Column",
        "Corrugated-Bulkhead",
    ]

    functions = [EP1, EP2, EP3, EP4, EP5, EP6]
    problems = [P1, P2, P3, P4, P5, P6]

    return names, functions, problems
