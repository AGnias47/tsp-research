#!/usr/bin/env python

import argparse

import optuna
from optuna.integration.mlflow import MLflowCallback

from src.algorithms.q_learning.double_q_learning import DoubleQLearning
from src.algorithms.q_learning.q_learning import QLearning
from src.utils.arg_parsing import get_filepath_for_problem


class QLearningObjective:
    def __init__(self, algorithm, problem):
        self.algorithm = algorithm
        self.problem = problem
        self.filepath = get_filepath_for_problem(problem)

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


def aco_objective():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--algorithm", required=True, choices=["as", "mmas", "q", "dq"]
    )
    parser.add_argument("-p", "--problem", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-n", "--trials", type=int)
    group.add_argument("-t", "--timeout", type=int)
    args = parser.parse_args()
    if args.trials:
        kwargs = {"n_trials": args.trials}
    elif args.timeout:
        kwargs = {"timeout": args.timeout}
    else:
        kwargs = {"n_trials": 10}
    if args.algorithm == "as":
        pass
    elif args.algorithm == "mmas":
        pass
    elif args.algorithm == "q":
        print("Running Q-Learning Study")
        study = optuna.create_study(study_name="Q-Learning Hyperparameter Tuning")
        try:
            study.optimize(
                func=QLearningObjective(QLearning, args.problem),
                n_jobs=2,
                callbacks=[MLflowCallback(metric_name="cost")],
                **kwargs
            )
        except KeyboardInterrupt:
            print("Ending study")
    elif args.algorithm == "dq":
        print("Running Double Q-Learning Study")
        study = optuna.create_study(study_name="Double Q-Learning Hyperparameter Tuning")
        try:
            study.optimize(
                func=QLearningObjective(DoubleQLearning, args.problem),
                n_jobs=1,
                callbacks=[MLflowCallback(metric_name="cost")],
                **kwargs
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
