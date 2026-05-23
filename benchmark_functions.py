from mealpy import FloatVar
import opfunu
from opfunu.cec_based import cec2020, cec2022


def classical_test_function():

    F1 = opfunu.name_based.a_func.Ackley02()
    L1 = F1.latex_formula
    P1 = {
        "bounds": FloatVar(lb=F1.lb, ub=F1.ub),
        "minmax": "min",
        "fit_func": F1.evaluate,
        "name": "Ackley2",
        "log_to": None,
        "save_population": False
    }

    F2 = opfunu.name_based.b_func.Brent()
    L2 = F2.latex_formula
    P2 = {
        "bounds": FloatVar(lb=F2.lb, ub=F2.ub),
        "minmax": "min",
        "fit_func": F2.evaluate,
        "name": "Brent",
        "log_to": None,
        "save_population": False
    }

    F3 = opfunu.name_based.c_func.ChungReynolds()
    L3 = F3.latex_formula
    P3 = {
        "bounds": FloatVar(lb=F3.lb, ub=F3.ub),
        "minmax": "min",
        "fit_func": F3.evaluate,
        "name": "ChungReynolds",
        "log_to": None,
        "save_population": False
    }

    F4 = opfunu.name_based.c_func.Cigar()
    L4 = F4.latex_formula
    P4 = {
        "bounds": FloatVar(lb=F4.lb, ub=F4.ub),
        "minmax": "min",
        "fit_func": F4.evaluate,
        "name": "Cigar",
        "log_to": None,
        "save_population": False
    }

    F5 = opfunu.name_based.m_func.Matyas()
    L5 = F5.latex_formula
    P5 = {
        "bounds": FloatVar(lb=F5.lb, ub=F5.ub),
        "minmax": "min",
        "fit_func": F5.evaluate,
        "name": "Matyas",
        "log_to": None,
        "save_population": False
    }

    F6 = opfunu.name_based.l_func.Leon()
    L6 = F6.latex_formula
    P6 = {
        "bounds": FloatVar(lb=F6.lb, ub=F6.ub),
        "minmax": "min",
        "fit_func": F6.evaluate,
        "name": "Leon",
        "log_to": None,
        "save_population": False
    }

    F7 = opfunu.name_based.m_func.Michalewicz()
    L7 = F7.latex_formula
    P7 = {
        "bounds": FloatVar(lb=F7.lb, ub=F7.ub),
        "minmax": "min",
        "fit_func": F7.evaluate,
        "name": "Michalewicz",
        "log_to": None,
        "save_population": False
    }

    F8 = opfunu.name_based.c_func.CrossInTray()
    L8 = F8.latex_formula
    P8 = {
        "bounds": FloatVar(lb=F8.lb, ub=F8.ub),
        "minmax": "min",
        "fit_func": F8.evaluate,
        "name": "CrossInTray",
        "log_to": None,
        "save_population": False
    }

    F9 = opfunu.name_based.h_func.Hosaki()
    L9 = F9.latex_formula
    P9 = {
        "bounds": FloatVar(lb=F9.lb, ub=F9.ub),
        "minmax": "min",
        "fit_func": F9.evaluate,
        "name": "Hosaki",
        "log_to": None,
        "save_population": False
    }

    F10 = opfunu.name_based.l_func.Langermann()
    L10 = F10.latex_formula
    P10 = {
        "bounds": FloatVar(lb=F10.lb, ub=F10.ub),
        "minmax": "min",
        "fit_func": F10.evaluate,
        "name": "Langermann",
        "log_to": None,
        "save_population": False
    }

    F11 = opfunu.name_based.l_func.Levy05()
    L11 = F11.latex_formula
    P11 = {
        "bounds": FloatVar(lb=F11.lb, ub=F11.ub),
        "minmax": "min",
        "fit_func": F11.evaluate,
        "name": "Levy5",
        "log_to": None,
        "save_population": False
    }

    F12 = opfunu.name_based.m_func.Mishra05()
    L12 = F12.latex_formula
    P12 = {
        "bounds": FloatVar(lb=F12.lb, ub=F12.ub),
        "minmax": "min",
        "fit_func": F12.evaluate,
        "name": "Mishra5",
        "log_to": None,
        "save_population": False
    }

    F13 = opfunu.name_based.a_func.Alpine02()
    L13 = F13.latex_formula
    P13 = {
        "bounds": FloatVar(lb=F13.lb, ub=F13.ub),
        "minmax": "min",
        "fit_func": F13.evaluate,
        "name": "Alpine2",
        "log_to": None,
        "save_population": False
    }

    F14 = opfunu.name_based.h_func.Hansen()
    L14 = F14.latex_formula
    P14 = {
        "bounds": FloatVar(lb=F14.lb, ub=F14.ub),
        "minmax": "min",
        "fit_func": F14.evaluate,
        "name": "Hansen",
        "log_to": None,
        "save_population": False
    }

    F15 = opfunu.name_based.h_func.Himmelblau()
    L15 = F15.latex_formula
    P15 = {
        "bounds": FloatVar(lb=F15.lb, ub=F15.ub),
        "minmax": "min",
        "fit_func": F15.evaluate,
        "name": "Himmelblau",
        "log_to": None,
        "save_population": False
    }

    F16 = opfunu.name_based.c_func.Chichinadze()
    L16 = F16.latex_formula
    P16 = {
        "bounds": FloatVar(lb=F16.lb, ub=F16.ub),
        "minmax": "min",
        "fit_func": F16.evaluate,
        "name": "Chichinadze",
        "log_to": None,
        "save_population": False
    }

    names = [
        "F1", "F2", "F3", "F4",
        "F5", "F6", "F7", "F8",
        "F9", "F10", "F11", "F12",
        "F13", "F14", "F15", "F16"
    ]

    functions = [
        F1, F2, F3, F4,
        F5, F6, F7, F8,
        F9, F10, F11, F12,
        F13, F14, F15, F16
    ]

    problems = [
        P1, P2, P3, P4,
        P5, P6, P7, P8,
        P9, P10, P11, P12,
        P13, P14, P15, P16
    ]

    latexs = [
        L1, L2, L3, L4,
        L5, L6, L7, L8,
        L9, L10, L11, L12,
        L13, L14, L15, L16
    ]

    return names, functions, problems, latexs


def cec20_test_function_with_bias(dimension=None):

    C1 = cec2020.F12020(ndim=dimension, f_bias=100)
    L1 = C1.latex_formula
    P1 = {
        "bounds": FloatVar(lb=C1.lb, ub=C1.ub),
        "minmax": "min",
        "obj_func": C1.evaluate,
        "name": "F12020",
        "log_to": None,
        "save_population": False
    }

    C2 = cec2020.F22020(ndim=dimension, f_bias=1100)
    L2 = C2.latex_formula
    P2 = {
        "bounds": FloatVar(lb=C2.lb, ub=C2.ub),
        "minmax": "min",
        "obj_func": C2.evaluate,
        "name": "F22020",
        "log_to": None,
        "save_population": False
    }

    C3 = cec2020.F32020(ndim=dimension, f_bias=700)
    L3 = C3.latex_formula
    P3 = {
        "bounds": FloatVar(lb=C3.lb, ub=C3.ub),
        "minmax": "min",
        "obj_func": C3.evaluate,
        "name": "F32020",
        "log_to": None,
        "save_population": False
    }

    C4 = cec2020.F42020(ndim=dimension, f_bias=1900)
    L4 = C4.latex_formula
    P4 = {
        "bounds": FloatVar(lb=C4.lb, ub=C4.ub),
        "minmax": "min",
        "obj_func": C4.evaluate,
        "name": "F42020",
        "log_to": None,
        "save_population": False
    }

    C5 = cec2020.F52020(ndim=dimension, f_bias=1700)
    L5 = C5.latex_formula
    P5 = {
        "bounds": FloatVar(lb=C5.lb, ub=C5.ub),
        "minmax": "min",
        "obj_func": C5.evaluate,
        "name": "F52020",
        "log_to": None,
        "save_population": False
    }

    C6 = cec2020.F62020(ndim=dimension, f_bias=1600)
    L6 = C6.latex_formula
    P6 = {
        "bounds": FloatVar(lb=C6.lb, ub=C6.ub),
        "minmax": "min",
        "obj_func": C6.evaluate,
        "name": "F62020",
        "log_to": None,
        "save_population": False
    }

    C7 = cec2020.F72020(ndim=dimension, f_bias=2100)
    L7 = C7.latex_formula
    P7 = {
        "bounds": FloatVar(lb=C7.lb, ub=C7.ub),
        "minmax": "min",
        "obj_func": C7.evaluate,
        "name": "F72020",
        "log_to": None,
        "save_population": False
    }

    C8 = cec2020.F82020(ndim=dimension, f_bias=2200)
    L8 = C8.latex_formula
    P8 = {
        "bounds": FloatVar(lb=C8.lb, ub=C8.ub),
        "minmax": "min",
        "obj_func": C8.evaluate,
        "name": "F82020",
        "log_to": None,
        "save_population": False
    }

    C9 = cec2020.F92020(ndim=dimension, f_bias=2400)
    L9 = C9.latex_formula
    P9 = {
        "bounds": FloatVar(lb=C9.lb, ub=C9.ub),
        "minmax": "min",
        "obj_func": C9.evaluate,
        "name": "F92020",
        "log_to": None,
        "save_population": False
    }

    C10 = cec2020.F102020(ndim=dimension, f_bias=2500)
    L10 = C10.latex_formula
    P10 = {
        "bounds": FloatVar(lb=C10.lb, ub=C10.ub),
        "minmax": "min",
        "obj_func": C10.evaluate,
        "name": "F102020",
        "log_to": None,
        "save_population": False
    }

    names = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]
    functions = [C1, C2, C3, C4, C5, C6, C7, C8, C9, C10]
    problems = [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10]
    latexs = [L1, L2, L3, L4, L5, L6, L7, L8, L9, L10]

    return names, functions, problems, latexs


def cec22_test_function_with_bias(dimension=None):

    C1 = cec2022.F12022(ndim=dimension, f_bias=300)
    L1 = C1.latex_formula
    P1 = {
        "bounds": FloatVar(lb=C1.lb, ub=C1.ub),
        "minmax": "min",
        "obj_func": C1.evaluate,
        "name": "F12022",
        "log_to": None,
        "save_population": False
    }

    C2 = cec2022.F22022(ndim=dimension, f_bias=400)
    L2 = C2.latex_formula
    P2 = {
        "bounds": FloatVar(lb=C2.lb, ub=C2.ub),
        "minmax": "min",
        "obj_func": C2.evaluate,
        "name": "F22022",
        "log_to": None,
        "save_population": False
    }

    C3 = cec2022.F32022(ndim=dimension, f_bias=600)
    L3 = C3.latex_formula
    P3 = {
        "bounds": FloatVar(lb=C3.lb, ub=C3.ub),
        "minmax": "min",
        "obj_func": C3.evaluate,
        "name": "F32022",
        "log_to": None,
        "save_population": False
    }

    C4 = cec2022.F42022(ndim=dimension, f_bias=800)
    L4 = C4.latex_formula
    P4 = {
        "bounds": FloatVar(lb=C4.lb, ub=C4.ub),
        "minmax": "min",
        "obj_func": C4.evaluate,
        "name": "F42022",
        "log_to": None,
        "save_population": False
    }

    C5 = cec2022.F52022(ndim=dimension, f_bias=900)
    L5 = C5.latex_formula
    P5 = {
        "bounds": FloatVar(lb=C5.lb, ub=C5.ub),
        "minmax": "min",
        "obj_func": C5.evaluate,
        "name": "F52022",
        "log_to": None,
        "save_population": False
    }

    C6 = cec2022.F62022(ndim=dimension, f_bias=1800)
    L6 = C6.latex_formula
    P6 = {
        "bounds": FloatVar(lb=C6.lb, ub=C6.ub),
        "minmax": "min",
        "obj_func": C6.evaluate,
        "name": "F62022",
        "log_to": None,
        "save_population": False
    }

    C7 = cec2022.F72022(ndim=dimension, f_bias=2000)
    L7 = C7.latex_formula
    P7 = {
        "bounds": FloatVar(lb=C7.lb, ub=C7.ub),
        "minmax": "min",
        "obj_func": C7.evaluate,
        "name": "F72022",
        "log_to": None,
        "save_population": False
    }

    C8 = cec2022.F82022(ndim=dimension, f_bias=2200)
    L8 = C8.latex_formula
    P8 = {
        "bounds": FloatVar(lb=C8.lb, ub=C8.ub),
        "minmax": "min",
        "obj_func": C8.evaluate,
        "name": "F82022",
        "log_to": None,
        "save_population": False
    }

    C9 = cec2022.F92022(ndim=dimension, f_bias=2300)
    L9 = C9.latex_formula
    P9 = {
        "bounds": FloatVar(lb=C9.lb, ub=C9.ub),
        "minmax": "min",
        "obj_func": C9.evaluate,
        "name": "F92022",
        "log_to": None,
        "save_population": False
    }

    C10 = cec2022.F102022(ndim=dimension, f_bias=2400)
    L10 = C10.latex_formula
    P10 = {
        "bounds": FloatVar(lb=C10.lb, ub=C10.ub),
        "minmax": "min",
        "obj_func": C10.evaluate,
        "name": "F102022",
        "log_to": None,
        "save_population": False
    }

    C11 = cec2022.F112022(ndim=dimension, f_bias=2600)
    L11 = C11.latex_formula
    P11 = {
        "bounds": FloatVar(lb=C11.lb, ub=C11.ub),
        "minmax": "min",
        "obj_func": C11.evaluate,
        "name": "F112022",
        "log_to": None,
        "save_population": False
    }

    C12 = cec2022.F122022(ndim=dimension, f_bias=2700)
    L12 = C12.latex_formula
    P12 = {
        "bounds": FloatVar(lb=C12.lb, ub=C12.ub),
        "minmax": "min",
        "obj_func": C12.evaluate,
        "name": "F122022",
        "log_to": None,
        "save_population": False
    }

    names = [
        "C1", "C2", "C3", "C4",
        "C5", "C6", "C7", "C8",
        "C9", "C10", "C11", "C12"
    ]

    functions = [
        C1, C2, C3, C4,
        C5, C6, C7, C8,
        C9, C10, C11, C12
    ]

    problems = [
        P1, P2, P3, P4,
        P5, P6, P7, P8,
        P9, P10, P11, P12
    ]

    latexs = [
        L1, L2, L3, L4,
        L5, L6, L7, L8,
        L9, L10, L11, L12
    ]

    return names, functions, problems, latexs
