import correctionlib
import correctionlib.schemav2 as schema
import rich
import json
import numpy as np
import gzip

comparison_operators = [">=", "<=", ">", "<"]

variable_information_dict = {
    "eta": {
        "description": "pseudorapidity of the jet",
        "type": "real",
        "range": (-5.2, 5.2),
    },
    "abs_eta": {
        "description": "absolute value of pseudorapidity of the jet",
        "type": "real",
        "range": (0.0, 5.2),
    },
    "chHEF": {
        "description": "charged Hadron Energy Fraction",
        "type": "real",
        "range": (-1.0, 1.0),
    },
    "neHEF": {
        "description": "neutral Hadron Energy Fraction",
        "type": "real",
        "range": (-1.0, 1.0),
    },
    "chEmEF": {
        "description": "charged Electromagnetic Energy Fraction",
        "type": "real",
        "range": (-1.0, 1.0),
    },
    "neEmEF": {
        "description": "neutral Electromagnetic Energy Fraction",
        "type": "real",
        "range": (-1.0, 1.0),
    },
    "muEF": {
        "description": "muon Energy Fraction",
        "type": "real",
        "range": (-1.0, 1.0),
    },
    "chMultiplicity": {
        "description": "charged Multiplicity",
        "type": "int",
        "range": (-1.0, 99999), # different syntax for infinity in correctionlib <2.6.0 and >=2.6.0 so use 99999 for now
        "flow": "error",
    },
    "neMultiplicity": {
        "description": "neutral Multiplicity",
        "type": "int",
        "range": (-1.0, 99999), # different syntax for infinity in correctionlib <2.6.0 and >=2.6.0 so use 99999 for now
        "flow": "error",
    },
    "multiplicity": {
        "description": "charged Multiplicity + neutral Multiplicity",
        "type": "int",
        "range": (-2.0, 99999), # different syntax for infinity in correctionlib <2.6.0 and >=2.6.0 so use 99999 for now
        "flow": "error",
    },
}

# eta and abs_eta should not be used together
variable_ordering = ["eta", "abs_eta", "chHEF", "neHEF", "chEmEF", "neEmEF", "muEF", "chMultiplicity", "neMultiplicity", "multiplicity"]

legacy_to_nanoaod_name_map = {
    "CHF": "chHEF",
    "NHF": "neHEF",
    "CEMF": "chEmEF",
    "NEMF": "neEmEF",
    "MUF": "muEF",
    "CHM": "chMultiplicity",
    "NumNeutralParticle": "neMultiplicity",
    "NumConst": "multiplicity",
}

# flow option to use
flow_option = "clamp" # use clamp to handle open bound on upperbound of binning node

# whether to fix input variables
# if this is false, will only use variables used in JetID criteria
# if this is true, will use 9 inputs:  "eta"/"abs_eta", "chHEF", "neHEF", "chEmEF", "neEmEF", "muEF", "chMultiplicity", "neMultiplicity", "multiplicity"
is_fixed_input_variables = True

# change +- np.inf to "+inf"/"-inf" string
def inf2str(value):
    if np.isposinf(value):
        return "+inf"
    elif np.isneginf(value):
        return "-inf"
    return value

# find the first comparison operator in the string expression
def find_comparison_operator(string_expression):
    for comparison_operator in comparison_operators:
        index = string_expression.find(comparison_operator)
        if index >= 0: # find
            return index, comparison_operator
    return -1, None

# split unit expression by comparison operator
# e.g. split "NHF>0.9" to "NHF", ">", "0.9"
def split_unit_expression(string_expression):
    # find a comparison operator
    # the expression should contain at least one comparison operator
    index0, comparison_operator0 = find_comparison_operator(string_expression)
    if index0 == -1:
        raise ValueError(f"Cannot parse comparison operators from {string_expression}: Missing comparison operators. Available operators are " + ", ".join(comparison_operators))
    
    # check if there is another comparison operator  
    index1, comparison_operator1 = find_comparison_operator(string_expression[index0+len(comparison_operator0):])
    if index1 != -1:
        index1 += (index0+len(comparison_operator0)) # add offset so index1 is absolute index for input string_expression
    else: # try to find also in the substring before first comparison operator
        index1, comparison_operator1 = find_comparison_operator(string_expression[:index0])
        if index1 != -1: # find another one, swap
            index0, index1 = index1, index0
            comparison_operator0, comparison_operator1 = comparison_operator1, comparison_operator0

    if index1 == -1: # there is no another comparison operator
        # this assumes var_name first, e.g. "NHF>0.9" and not "0.9<NHF"
        var_name = string_expression[:index0]
        var_threshold_value = float(string_expression[index0+len(comparison_operator0):])
        return var_name, comparison_operator0, var_threshold_value
    else: # there is another comparison operator
        # this assumes var_name in the middle, e.g. "0<=NHF<0.9"
        var_name = string_expression[index0+len(comparison_operator0):index1]
        var_threshold_value0 = float(string_expression[:index0])
        var_threshold_value1 = float(string_expression[index1+len(comparison_operator1):])
        # try to check for error
        if ((comparison_operator0 == "<" or comparison_operator0 == "<=") and (comparison_operator1 == "<" or comparison_operator1 == "<=") and (var_threshold_value0 < var_threshold_value1)) \
                or ((comparison_operator0 == ">" or comparison_operator0 == ">=") and (comparison_operator1 == ">" or comparison_operator1 == ">=") and (var_threshold_value0 > var_threshold_value1)):    
            return var_name, (comparison_operator0, comparison_operator1), (var_threshold_value0, var_threshold_value1)
        else:
            raise ValueError(f"Cannot parse comparison operators from {string_expression}: Invalid expression with two comparison operators. Expression should be in the form number </<= var </<= number or number >/>= var >/>= number.")

# remove extra white spaces and add spaces properly
def prettify_expression(string_expression):
    string_expression = string_expression.replace(" ", "") # remove spaces
    return " && ".join(string_expression.split("&&"))

# change legacy naming to branch names in nanoaod
def convert_legacy_to_nanoaod_name(string_expression):
    for miniaod_name, nanoaod_name in legacy_to_nanoaod_name_map.items():
        string_expression = string_expression.replace(miniaod_name, nanoaod_name)
    return string_expression

# sort unit expression (NHF>0.9) in full expression in specified order
def sort_expression(string_expression):
    var_name_to_expr_map = dict()
    for unit_str_expr in string_expression.split("&&"):
        var_name, *_ = split_unit_expression(unit_str_expr.strip())
        var_name_to_expr_map[var_name] = unit_str_expr.strip()
    sorted_unit_str_expr = list()
    for name in variable_ordering:
        if name in var_name_to_expr_map:
            sorted_unit_str_expr.append(var_name_to_expr_map[name])
    return " && ".join([var_name_to_expr_map[name] for name in variable_ordering if name in var_name_to_expr_map])

# combine prettify, conversion to branch names in nanoaod, and sort
def preprocess_expression(string_expression):
    string_expression = prettify_expression(string_expression)
    string_expression = convert_legacy_to_nanoaod_name(string_expression)
    string_expression = sort_expression(string_expression)
    return string_expression

# adjust threshold for int variable because bins are defined as [low, high)
def adjust_threshold_for_int_variable(comparison_operator, var_threshold_value):
    if comparison_operator == ">":
        comparison_operator = ">="
        var_threshold_value += 1
    elif comparison_operator == "<=":
        comparison_operator = "<"
        var_threshold_value += 1
    return comparison_operator, var_threshold_value

# create a correction for unit expression with one comparison operator, e.g. NHF<0.9
def create_unit_jetId_correction_one_comparison_operator(var_name, comparison_operator, var_threshold_value, valid_content=1):
    # adjust threshold for int variable because bins are defined as [low, high)
    if variable_information_dict[var_name]["type"] == "int":
        comparison_operator, var_threshold_value = adjust_threshold_for_int_variable(comparison_operator, var_threshold_value)

    # retrieve min and max value for a variable
    var_min_value, var_max_value = variable_information_dict[var_name]["range"]
    if var_threshold_value < var_min_value or var_threshold_value > var_max_value:
        raise ValueError(f"Threshold value must be in between allowed values ({var_min_value}, {var_max_value}) of a variable {var_name}") 

    # fill bin content
    content = [0.0, 0.0]
    if comparison_operator == "<" or comparison_operator == "<=":
        content = [valid_content, 0.0]
    elif comparison_operator == ">" or comparison_operator == ">=":
        content = [0.0, valid_content]

    return schema.Binning(
        nodetype = "binning",
        input = var_name,
        edges = [inf2str(var_min_value), var_threshold_value, inf2str(var_max_value)],
        flow = variable_information_dict[var_name].get("flow", flow_option),
        content = content,
    )

# create a correction for unit expression with two comparison operators, e.g. 0<=NHF<0.9
def create_unit_jetId_correction_two_comparison_operators(var_name, comparison_operator, var_threshold_value, valid_content=1):
    comparison_operator0, comparison_operator1 = comparison_operator
    var_threshold_value0, var_threshold_value1 = var_threshold_value

    # swap
    if var_threshold_value0 > var_threshold_value1:
        var_threshold_value0, var_threshold_value1 = var_threshold_value1, var_threshold_value0
        comparison_operator0, comparison_operator1 = comparison_operator1, comparison_operator0
        comparison_operator0 = comparison_operator0.replace(">", "<")
        comparison_operator1 = comparison_operator1.replace(">", "<")

    # adjust threshold for int variable because bins are defined as [low, high)
    if variable_information_dict[var_name]["type"] == "int":
        comparison_operator0, var_threshold_value0 = adjust_threshold_for_int_variable(comparison_operator0, var_threshold_value0)
        comparison_operator1, var_threshold_value1 = adjust_threshold_for_int_variable(comparison_operator1, var_threshold_value1)

    # retrieve min and max value for a variable
    var_min_value, var_max_value = variable_information_dict[var_name]["range"]
    if var_threshold_value0 < var_min_value or var_threshold_value1 < var_min_value or var_threshold_value0 > var_max_value or var_threshold_value1 > var_max_value:
        raise ValueError(f"Threshold value must be in between allowed values ({var_min_value}, {var_max_value}) of a variable {var_name}")

    # fill bin content
    content = [0.0, valid_content, 0.0]

    return schema.Binning(
        nodetype = "binning",
        input = var_name,
        edges = [inf2str(var_min_value), var_threshold_value0, var_threshold_value1, inf2str(var_max_value)],
        flow = variable_information_dict[var_name].get("flow", flow_option),
        content = content,
    )

# create a correction for unit expression
def create_unit_jetId_correction(string_expression, valid_content=1):
    var_name, comparison_operator, var_threshold_value = split_unit_expression(string_expression)
    if isinstance(comparison_operator, str):
        return create_unit_jetId_correction_one_comparison_operator(var_name, comparison_operator, var_threshold_value, valid_content)
    else: 
        return create_unit_jetId_correction_two_comparison_operators(var_name, comparison_operator, var_threshold_value, valid_content)

# add the condition >= 0 to the expression
def add_nonnegative_condition_to_expression(string_expression):
    var_name, comparison_operator, var_threshold_value = split_unit_expression(string_expression)
    if isinstance(comparison_operator, str): # one comparison operator
        if comparison_operator == "<" or comparison_operator == "<=": # prepend "0<=" 
            return f"0<={string_expression}"
        elif comparison_operator == ">" or comparison_operator == ">=":
            if var_threshold_value < 0: # if threshold is < 0, change threshold to 0
                return f"{var_name}{comparison_operator}0"
            else:
                return string_expression
        else:
            raise ValueError(f"Unsupported comparison operator in {string_expression}. Supported operators are: <, <=, >, >=.")
    else:
        comparison_operator0, comparison_operator1 = comparison_operator
        var_threshold_value0, var_threshold_value1 = var_threshold_value
        if comparison_operator0 == "<" or comparison_operator0 == "<=":
            if var_threshold_value0 < 0:
                return f"0<={var_name}{comparison_operator1}{var_threshold_value1}"
            else:
                return string_expression
        elif comparison_operator0 == ">" or comparison_operator0 == ">=":
            if var_threshold_value1 < 0:
                return f"{var_threshold_value0}{comparison_operator0}{var_name}>=0"
            else:
                return string_expression
        else:
            raise ValueError(f"Unsupported comparison operator in {string_expression}. Supported operators are: <, <=, >, >=.")

# create a JetID correction for one working point (e.g. Tight, TightLeptonVeto) and one era
def create_jetId_correction(eta_bin_edges, criteria_expressions,
                            input_abs_eta=False, # whether to use eta (False) or absolute eta (True) as input. If using eta (False), transform node to apply absolute value will be added 
                            name=None, description=None, version=1, verbose_description=False):
    assert len(eta_bin_edges)-1 == len(criteria_expressions), f"Mismatch number of eta bins ({len(eta_bin_edges)-1}) and criteria expression (len(criteria_expressions))"
    
    def create_jetId_correction_per_eta_bin(string_expression):
        string_expression = string_expression.replace(" ", "") # remove spaces
        inputs = list()
        content = 1
        for unit_str_expr in string_expression.split("&&")[::-1]:
            var_name, *_ = split_unit_expression(unit_str_expr)
            if var_name in ["chHEF", "neHEF", "chEmEF", "neEmEF", "muEF", "chMultiplicity", "neMultiplicity", "multiplicity"]:
                unit_str_expr = add_nonnegative_condition_to_expression(unit_str_expr) # add the condition >=0 to the expression 
            content = create_unit_jetId_correction(unit_str_expr, valid_content=content)
            inputs.append(content.input)
        return inputs, content
    
    criteria_inputs = list()
    if not input_abs_eta: # use eta as input
        criteria_inputs.append("eta")
    else: # use abs(eta) as input
        criteria_inputs.append("abs_eta")
    criteria_corrections = list()
    for expr in criteria_expressions:
        inputs, correction = create_jetId_correction_per_eta_bin(expr)
        criteria_inputs += inputs
        criteria_corrections.append(correction)
    input_names = list()
    if is_fixed_input_variables:
        input_names = ["eta"]
        if input_abs_eta:
            input_names = ["abs_eta"]
        input_names += ["chHEF", "neHEF", "chEmEF", "neEmEF", "muEF", "chMultiplicity", "neMultiplicity", "multiplicity"]
    else:
        input_names = [input_name for input_name in variable_ordering if input_name in criteria_inputs]
    inputs = [schema.Variable(name=input_name, type=variable_information_dict[input_name]["type"], description=variable_information_dict[input_name]["description"])
              for input_name in input_names]
    if verbose_description:
        for eta_bin_idx in range(len(eta_bin_edges)-1):
            description += "\n" + f"abs(eta) in [{eta_bin_edges[eta_bin_idx]}, {eta_bin_edges[eta_bin_idx+1]}): " + criteria_expressions[eta_bin_idx]

    data = None
    if not input_abs_eta: # use eta as input
        data = schema.Transform(
            nodetype = "transform",
            input = "eta",
            rule = schema.Formula(
                nodetype = "formula",
                variables = ["eta"],
                parser = "TFormula",
                expression = "abs(x)",
            ),
            content = schema.Binning(
                nodetype = "binning",
                input = "eta",
                edges = eta_bin_edges,
                flow = variable_information_dict["eta"].get("flow", flow_option),
                content = criteria_corrections
            )
        )
    else: # use abs(eta) as input
        data = schema.Binning(
            nodetype = "binning",
            input = "abs(eta)",
            edges = eta_bin_edges,
            flow = variable_information_dict["abs_eta"].get("flow", flow_option),
            content = criteria_corrections
        )

    return schema.Correction(
        name = name,
        description = description,
        version = version,
        inputs = inputs,
        output = schema.Variable(name="jet id", type="real", description="jet identification"),
        data = data
    )

# create a JetID correctionSet for multiple working point (e.g. Tight, TightLeptonVeto) and multiple eras
# by combining JetID corrections for each working point and each era
def create_jetId_correctionSet(criteria_tasks, description="", input_abs_eta=False):
    corrections = list()
    for criteria_task in criteria_tasks:
        criteria_task["criteria_expressions"] = [preprocess_expression(_) for _ in criteria_task["criteria_expressions"]]
        corrections.append(create_jetId_correction(**criteria_task, input_abs_eta=input_abs_eta))
    return schema.CorrectionSet(
        schema_version = 2,
        description = description,
        corrections = corrections,
    )

# write correctionSet to a json file
def write_correctionSet(correctionSet, filename, write_gzip=False):
    with open(filename, "w") as fout:
        json.dump(correctionSet.dict(exclude_unset=True), fout, indent=4)
    if write_gzip:
        with gzip.open(filename+".gz", "wt") as fout:
            json.dump(correctionSet.dict(exclude_unset=True), fout)
