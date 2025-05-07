"""
Evaluation module providing basic metrics to run and analyze GLocalX's results.
Two evaluators are provided, DummyEvaluator, which does not optimize performance (stored in base_evaluators),
and MemEvaluator, which stores previously computed measures to speed-up performance.
"""
from abc import abstractmethod

import numpy as np
from scipy.spatial.distance import hamming

import logzero

from src.utils.coverage_utils import binary_fidelity_old, covers, coverage_matrix, coverage_size


class Evaluator:
    """Evaluator interface. Evaluator objects provide coverage and fidelity_weight utilities."""

    @abstractmethod
    def coverage(self, rules, patterns, target=None, ids=None):
        """Compute the coverage of @rules over @patterns.
        Args:
            rules (list) or (Rule):
            patterns (numpy.array): The validation set.
            target (numpy.array): The labels, if any. None otherwise. Defaults to None.
            ids (numpy.array): Unique identifiers to tell each element in @patterns apart.
        Returns:
            numpy.array: The coverage matrix.
        """
        pass

    @abstractmethod
    def coverage_size(self, rule, x, ids=None):
        """Evaluate the cardinality of the coverage of unit on c.

        Args:
            rule (Rule): The rule.
            x (numpy.array): The validation set.
            ids (numpy.array): Unique identifiers to tell each element in @patterns apart.
        Returns:
            int: Number of records of X covered by rule.
        """
        pass

    @abstractmethod
    def binary_fidelity(self, unit, x, y, ids=None, default=np.nan):
        """Evaluate the goodness of unit.
        Args:
            unit (Unit): The unit to evaluate.
            x (numpy.array): The data.
            y (numpy.array): The labels.
            ids (numpy.array): Unique identifiers to tell each element in @patterns apart.
            default (int): Default prediction when no rule covers a record.
        Returns:
              float: The unit's fidelity_weight
        """
        pass

    @abstractmethod
    def binary_fidelity_model(self, units, x, y, k=1, default=None, ids=None):
        """Evaluate the goodness of the `units`.
        Args:
            units (Union(list, set)): The units to evaluate.
            x (numpy.array): The data.
            y (numpy.array): The labels.
            k (int): Number of rules to use in the Laplacian prediction schema.
            default (int): Default prediction for records not covered by the unit.
            ids (numpy.array): Unique identifiers to tell each element in @c apart.
        Returns:
            float: The units fidelity_weight.
        """
        pass

    @abstractmethod
    def covers(self, rule, x):
        """Does @rule cover c?

        Args:
            rule (Rule): The rule.
            x (numpy.array): The record.
        Returns:
            bool: True if this rule covers c, False otherwise.
        """
        pass

    @abstractmethod
    def bic(self, rules, vl, fidelity_weight=1., complexity_weight=1.):
        """
        Compute the Bayesian Information Criterion for the given `rules` set.
        Args:
            rules (set): Ruleset.
            vl (numpy.array): Validation set.
            fidelity_weight (float): Weight to fidelity_weight (BIC-wise).
            complexity_weight (float): Weight to complexity_weight (BIC-wise).
        Returns:
            tuple: Triple (BIC, log likelihood, complexity_weight).
        """
        pass

# Copied from https://github.com/Dudnik-Ilia/glocalx.
# Modified by kodaiyano.
class MemEvaluator(Evaluator):
    """Memoization Evaluator to avoid evaluating the same measures over the same data."""

    def __init__(self, model_ai, fidelity_weight:float, complexity_weight:float):
        # AI model we try to mimic
        self.model_ai = model_ai
        self.fidelity_weight = fidelity_weight
        self.complexity_weight = complexity_weight
        # For memoization
        self._coverages = dict()
        self._intersecting = dict()
        self._like_compl = dict()
        self._distances = dict()
        self._binary_fidelities = dict()
        # For storing train dataset
        self._x = None
        self._y = None
        self._train_set = None

    def coverage(self, rules, x=None, y=None):
        """Compute the coverage of rules over samples.
        Args:
            rules (Union(Rule, list): Rule (or list of rules) whose coverage to compute.
            x (numpy.array): several examples.
            y (numpy.array): The labels, if any.
        Returns:
            numpy.array: The coverage matrix.
        """
        if x is None:
            # If nothing was passed, use the training data
            x = self._x
            y = self._y

        rules_ = [rules] if not isinstance(rules, list) and not isinstance(rules, set) else rules

        # memoization hash table
        mem = self._coverages
        for rule in rules_:
            if rule not in mem:
                mem[rule] = coverage_matrix(rule, x, y)
        cov = np.array([mem[rule] for rule in rules_])

        return cov

    def distance(self, A, B):
        """
        Compute the distance between ruleset `A` and ruleset `B`.
        Using Memoization for remembering distances
        Args:
            A (iterable): Ruleset.
            B (iterable): Ruleset.
        Returns:
            (float): The Jaccard distance between the two.
        """
        # If A ruleset already was calculated with respect to B
        if tuple(A) in self._distances and tuple(B) in self._distances[tuple(A)]:
            diff = self._distances[tuple(A)][tuple(B)]
            return diff
        # Or B to A
        if tuple(B) in self._distances and tuple(A) in self._distances[tuple(B)]:
            diff = self._distances[tuple(B)][tuple(A)]
            return diff

        # New distance Compute
        coverage_A = self.coverage(A).sum(axis=0)
        coverage_B = self.coverage(B).sum(axis=0)
        diff = hamming(coverage_A, coverage_B)

        # Saving the results
        # If A/B already was in self.distances
        if tuple(A) in self._distances:
            self._distances[tuple(A)][tuple(B)] = diff
        if tuple(B) in self._distances:
            self._distances[tuple(B)][tuple(A)] = diff

        # If it is 1st time for A/B
        if tuple(A) not in self._distances:
            self._distances[tuple(A)] = {tuple(B): diff}
        if tuple(B) not in self._distances:
            self._distances[tuple(B)] = {tuple(A): diff}

        return diff

    def binary_fidelity(self, rule, x=None, y=None):
        """
        Evaluate the goodness of rule via "1-hamming distance" -> the higher the better. \n
        Note: Compaired to the old version, here we do not take into account default prediction for all the other samples, which are not covered by the rule
        Args:
            rule (Unit): The unit to evaluate.
            x (numpy.array): The data. Training data is used if None
            y (numpy.array): The labels. Training data is used if None
        Returns:
              float: The rule's fidelity.
        """
        if x is None:
            # Use training data
            x, y = self._x, self._y

        if rule not in self._binary_fidelities:
            # Calculate coverage of the rule (ids that are under rule's scope)
            coverage = self.coverage(rule, x).flatten()
            covered_indices = np.where(coverage)[0]
            if len(covered_indices) == 0:
                return 0
            
            # Predictions for covered samples according to the rule
            unit_predictions = np.full(len(covered_indices), rule.consequence)
            
            y = y.squeeze()
            covered_y = y[covered_indices]
            
            # Calculate the fidelity as 1 - Hamming distance for covered samples
            hamming_distance = hamming(unit_predictions, covered_y)
            fidelity = 1 - hamming_distance
            # Save
            self._binary_fidelities[rule] = fidelity

        return self._binary_fidelities[rule]

    def binary_fidelity_model(self, rules, x=None, y=None, k=1, default=None):
        """Calculate the Log-Likelyhood (fidelity) of the `rules`.
        Args:
            rules (Union(list, set)): The rules to evaluate.
            x (numpy.array): The data. Training data is used if None
            y (numpy.array): The labels. Training data is used if None
            k (int): Number of rules to use in the Laplacian prediction schema.
            default (int): Default prediction for records not covered by the unit.
        Returns:
              float: fidelity of rules as a model.
        """
        if x is None:
            # Use training data
            x, y = self._x, self._y
        
        y = y.squeeze()
        
        if default == None:
            default = int(y.mean().round())

        # fidelity for each rule
        fidelities = np.array([self.binary_fidelity(rule, x, y) for rule in rules])
        # True/false mask for each rule whether it covers sample
        coverage = self.coverage(rules, x)

        if len(rules) == 0:
            predictions = [default] * len(y)
        else:
            rules_consequences = np.array([r.consequence for r in rules])
            # Fast computation for k = 1
            if k == 1:
                # Coverage matrix (masks), where we have fidelities (if True) and 0 (if False)
                weighted_coverage_scores = coverage * fidelities.reshape(-1, 1)
                
                # Take the best fidelity if several rules apply for the same sample -> get index of best rule
                best_rule_per_record_idx = weighted_coverage_scores.argmax(axis=0).squeeze()
                assert len(best_rule_per_record_idx) == len(x)
                predictions = rules_consequences[best_rule_per_record_idx]
                
                # Replace predictions of non-covered records with default prediction
                predictions[coverage.sum(axis=0) == 0] = default
            # Iterative computation
            else:
                predictions = []
                for record in range(len(x)):
                    record_coverage = np.argwhere(coverage[:, record]).ravel()
                    if len(record_coverage) == 0:
                        prediction = default
                    else:
                        rules_for_0 = record_coverage[rules_consequences[record_coverage] == 0]
                        rules_for_1 = record_coverage[rules_consequences[record_coverage] == 1]
                        fid_0 = fidelities[rules_for_0]
                        fid_1 = fidelities[rules_for_1]
                        argsort_scores_0 = np.flip(np.argsort(fidelities[rules_for_0])[-k:])
                        argsort_scores_1 = np.flip(np.argsort(fidelities[rules_for_1])[-k:])
                        top_scores_0 = fid_0[argsort_scores_0]
                        top_scores_1 = fid_1[argsort_scores_1]

                        if len(top_scores_0) == 0 and len(top_scores_1) > 0:
                            prediction = 1
                        elif len(top_scores_1) == 0 and len(top_scores_0) > 0:
                            prediction = 0
                        elif len(top_scores_1) == 0 and len(top_scores_0) == 0:
                            prediction = default
                        else:
                            prediction = 0 if np.mean(top_scores_0) > np.mean(top_scores_1) else 1

                    predictions.append(prediction)
                predictions = np.array(predictions)
        fidelity = 1 - hamming(predictions, y) if len(y) > 0 else 0

        return fidelity

    def bic(self, rules, test_data=None, logging_msg=''):
        """
        Compute the Bayesian Information Criterion for the given `rules` set.
        Args:
            rules (set): Ruleset.
            test_data (numpy.array): Validation set. If None, then training data is used.
            logging_msg (str): used for logging purposes
        Returns:
            float: BIC of a model 
        """
        # If already calculated (with training)
        if tuple(rules) in self._like_compl and not test_data:
            log_likelihood, model_complexity = self._like_compl[tuple(rules)]
            n, m = self._x.shape
        else:
            if not test_data:
                x, y = self._x, self._y
            else:
                x, y = test_data[:, :-1], test_data[:, -1]
            n, m = x.shape
            default = int(y.mean().round())
            log_likelihood = self.binary_fidelity_model(rules, x, y, default=default)
            model_complexity = np.mean([len(r) / m for r in rules])
            # Save
            self._like_compl[tuple(rules)] = (log_likelihood, model_complexity)

        model_bic = - (self.fidelity_weight * log_likelihood - self.complexity_weight * model_complexity / n)

        # Log calculation or retreival
        logzero.logger.debug(f"\t{logging_msg}| Log-like: {log_likelihood:.4f} Complexity: {model_complexity:.4f}")

        return model_bic

    def forget(self, rules, A=None, B=None):
        """
        Remove rules from this Evaluator's memory. Return the updated evaluator.
        Args:
            rules (iterable): Rules to remove.
            A (set): Rules merged.
            B (set): Rules merged.
        Returns:
            MemEvaluator: This evaluator with no memory of `rules`.

        """
        for rule in rules:
            if rule in self._binary_fidelities:
                del self._binary_fidelities[rule]
            if rule in self._coverages:
                del self._coverages[rule]

        if A is not None and B is not None:
            # Delete the whole A, as it has been merged and does not exist anymore
            del self._distances[tuple(A)]
            # Delete the whole B, as it has been merged and does not exist anymore
            del self._distances[tuple(B)]
            # Delete every reference to any of them, as they have been merged and do not exist anymore
            for T in self._distances:
                if tuple(A) in self._distances[T]:
                    del self._distances[T][tuple(A)]
                if tuple(B) in self._distances[T]:
                    del self._distances[T][tuple(B)]

        return self