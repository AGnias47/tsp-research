#!/usr/bin/env python

import argparse

import optuna
from optuna.integration.mlflow import MLflowCallback
from src.algorithms.aco.ant_system import AntSystem
from src.algorithms.aco.max_min_ant_system import MaxMinAntSystem
from src.algorithms.q_learning.double_q_learning import DoubleQLearning
from src.algorithms.q_learning.q_learning import QLearning
from src.utils.arg_parsing import get_filepath_for_problem

DEFAULT_TRIALS = 10


class Objective:
    def __init__(self, algorithm, problem):
        self.algorithm = algorithm
        self.problem = problem
        self.filepath = get_filepath_for_problem(problem)


class ACOObjective(Objective):
    def __init__(self, algorithm, problem, mmas=False):
        super().__init__(algorithm, problem)
        self.mmas = mmas

    def __call__(self, trial: optuna.trial.BaseTrial):
        trial.set_user_attr("problem", self.problem)
        alpha = trial.suggest_int("alpha", 1, 2)
        beta = trial.suggest_int("beta", 2, 5)
        iterations = trial.suggest_int("iterations", 100, 10000)
        if self.mmas:
            rho = trial.suggest_float("rho", 0.01, 0.2)
            st = trial.suggest_int("stagnation_tolerance", 20, 350)
            kwargs = {"rho": rho, "stagnation_tolerance": st}
        else:
            rho = trial.suggest_float("rho", 0.3, 0.7)
            kwargs = {"rho": rho}
        solver = self.algorithm(
            filepath=self.filepath,
            alpha=alpha,
            beta=beta,
            iterations=iterations,
            **kwargs,
        )
        (cost, route), total_time = solver.run_tsp()
        return cost


class QLearningObjective(Objective):
    def __init__(self, algorithm, problem):
        super().__init__(algorithm, problem)

    def __call__(self, trial: optuna.trial.BaseTrial):
        trial.set_user_attr("problem", self.problem)
        alpha = trial.suggest_float("alpha", 0.0001, 0.1)
        gamma = trial.suggest_float("gamma", 0.0001, 0.99999)
        epsilon = trial.suggest_categorical("epsilon", ["e1", "e2", "e3", "e4"])
        reward = trial.suggest_categorical("reward", ["r1", "r2", "r3"])
        solver = self.algorithm(
            filepath=self.filepath,
            alpha=alpha,
            gamma=gamma,
            epsilon_func_key=epsilon,
            reward_func_key=reward,
        )
        (cost, route), total_time = solver.run_tsp()
        return cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--algorithm", required=True, choices=["as", "mmas", "q", "dq"]
    )
    parser.add_argument("-p", "--problem", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-n",
        "--trials",
        type=int,
        help="Number of trials to run. Cannot be specified with timeout. "
        f"If neither are specified, {DEFAULT_TRIALS} trials run.",
    )
    group.add_argument(
        "-t",
        "--timeout",
        type=int,
        help="Time to run before stopping. Cannot be specified with trials. "
        f"If neither are specified, {DEFAULT_TRIALS} trials run.",
    )
    args = parser.parse_args()
    if args.trials:
        kwargs = {"n_trials": args.trials}
    elif args.timeout:
        kwargs = {"timeout": args.timeout}
    else:
        kwargs = {"n_trials": DEFAULT_TRIALS}
    if args.algorithm == "as":
        print("Running Ant System Study")
        study = optuna.create_study(study_name="Ant System Hyperparameter Tuning")
        try:
            study.optimize(
                func=ACOObjective(AntSystem, args.problem),
                n_jobs=2,
                callbacks=[MLflowCallback(metric_name="cost")],
                **kwargs,
            )
        except KeyboardInterrupt:
            print("Ending study")
    elif args.algorithm == "mmas":
        print("Running Max-Min Ant System Study")
        study = optuna.create_study(study_name="Max-Min Ant System Hyperparameter Tuning")
        try:
            study.optimize(
                func=ACOObjective(MaxMinAntSystem, args.problem, mmas=True),
                n_jobs=2,
                callbacks=[MLflowCallback(metric_name="cost")],
                **kwargs,
            )
        except KeyboardInterrupt:
            print("Ending study")
    elif args.algorithm == "q":
        print("Running Q-Learning Study")
        study = optuna.create_study(study_name="Q-Learning Hyperparameter Tuning")
        try:
            study.optimize(
                func=QLearningObjective(QLearning, args.problem),
                n_jobs=2,
                callbacks=[MLflowCallback(metric_name="cost")],
                **kwargs,
            )
        except KeyboardInterrupt:
            print("Ending study")
    elif args.algorithm == "dq":
        print("Running Double Q-Learning Study")
        study = optuna.create_study(
            study_name="Double Q-Learning Hyperparameter Tuning"
        )
        try:
            study.optimize(
                func=QLearningObjective(DoubleQLearning, args.problem),
                n_jobs=1,
                callbacks=[MLflowCallback(metric_name="cost")],
                **kwargs,
            )
        except KeyboardInterrupt:
            print("Ending study")
    else:
        raise ValueError("Invalid algorithm specified")
    print("Optuna study best trial:")
    trial = study.best_trial
    cost = trial.value
    print(f"Cost: {cost}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")
