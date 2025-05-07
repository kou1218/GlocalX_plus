import json
from numpy import inf
from .rule_glocalx import Rule
import pickle

def lore_to_glocalx(loaded_rules:list[str], feature_names:list, class_values:list) -> list:
    """Load LORE rules and convert it to GlocalX rules
    Args:
        lore_rules_file (str): Path to the Pickle file with LORE rules.
        info_file (str): Path to the info file containing the rules' metadata.
    Returns:
        (list): List of `Rule` objects.
    """

    lore_rules = [r for r in loaded_rules if len(r) > 0]
    output_rules = []
    
    for lore_rule in lore_rules:
        consequence = class_values.index(lore_rule.cons)
        premises = lore_rule.premises
        features = [feature_names.index(premise.att) for premise in premises]
        # print(f"features: {features}")
        # print(f"premises: {premises}")
        # print(f"consequence: {consequence}")
        ops = [premise.op for premise in premises]
        # print(f"ops: {ops}")
        values = [premise.thr for premise in premises]
        # print(f"values: {values}")
        values_per_feature = {feature: [val for f, val in zip(features, values) if f == feature]
                              for feature in features}
        # print(f"values_per_feature: {values_per_feature}")
        ops_per_feature = {feature: [op for f, op in zip(features, ops) if f == feature]
                           for feature in features}
        # print(f"ops_per_feature: {ops_per_feature}")

        output_premises = {}
        for f in features:
            values, operators = values_per_feature[f], ops_per_feature[f]
            # 1 value, either <= or >
            if len(values) == 1:
                if operators[0] == '<=':
                    output_premises[f] = (-inf, values[0])
                else:
                    output_premises[f] = (values[0], +inf)
            # 2 values, < x <=
            else:
                output_premises[f] = (min(values), max(values))
        
        print(f"output_premises: {output_premises}")

        transformed_rule = Rule(premises=output_premises, consequence=consequence, names=feature_names)
        print(f"transformed_rule: {transformed_rule}")
        # exit()
        output_rules.append(transformed_rule)

    output_rules = list(set(output_rules))

    return output_rules